use std::fmt::Write;
use std::sync::{Arc, Weak};

use anyhow::Result;
use codex_rmcp_client::SamplingHandler;
use futures::StreamExt;
use tokio::sync::RwLock;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::auth::CodexAuth;
use crate::client::ModelClient;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::codex::Session;
use crate::codex::SessionSettingsUpdate;
use crate::codex::TurnContext;
use crate::config::Config;
use crate::mcp_tool_call::handle_mcp_tool_call;
use crate::model_family::derive_default_model_family;
use crate::model_family::find_family_for_model;
use crate::protocol::AgentMessageEvent;
use crate::protocol::Event;
use crate::protocol::EventMsg;
use crate::tools::ToolRouter;
use crate::tools::spec::ToolsConfig;
use crate::tools::spec::ToolsConfigParams;
use crate::AuthManager;
use codex_protocol::ConversationId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;

/// Implementation of SamplingHandler that creates independent LLM calls
/// for MCP sampling requests, respecting the request's systemPrompt and
/// model preferences rather than reusing the session's ModelClient.
pub struct CodexSamplingHandler {
    config: RwLock<Option<Arc<Config>>>,
    auth_manager: RwLock<Option<Arc<AuthManager>>>,
    session: RwLock<Option<Weak<Session>>>,
}

impl Default for CodexSamplingHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexSamplingHandler {
    pub fn new() -> Self {
        Self {
            config: RwLock::new(None),
            auth_manager: RwLock::new(None),
            session: RwLock::new(None),
        }
    }

    /// Set the Config to use for sampling requests.
    /// This must be called before any sampling requests are made.
    pub async fn set_config(
        &self,
        config: Arc<Config>,
        auth_manager: Option<Arc<AuthManager>>,
    ) {
        *self.config.write().await = Some(config);
        *self.auth_manager.write().await = auth_manager;
    }

    /// Attach the active Codex session so sampling requests surface in the UI.
    pub async fn set_session(&self, session: Option<Arc<Session>>) {
        let weak = session.map(|s| Arc::downgrade(&s));
        *self.session.write().await = weak;
    }

    /// Get a clone of the config if it has been initialized.
    async fn get_config(&self) -> Result<Arc<Config>, rmcp::ErrorData> {
        self.config.read().await.clone().ok_or_else(|| {
            warn!("Sampling requested but Config not yet initialized");
            rmcp::ErrorData::internal_error("Config not initialized for sampling", None)
        })
    }
}

#[async_trait::async_trait]
impl SamplingHandler for CodexSamplingHandler {
    async fn create_message(
        &self,
        params: rmcp::model::CreateMessageRequestParam,
    ) -> Result<rmcp::model::CreateMessageResult, rmcp::ErrorData> {
        use rmcp::model::Content;
        use rmcp::model::CreateMessageResult;
        use rmcp::model::Role;
        use rmcp::model::SamplingMessage;

        info!(
            "Processing MCP sampling request with {} messages",
            params.messages.len()
        );

        let sampling_sub_id = format!("mcp-sampling-{}", ConversationId::new());
        self.notify_sampling_request(&params, &sampling_sub_id).await;

        let config = self.get_config().await?;
        let auth_manager = { self.auth_manager.read().await.clone() };
        let auth = auth_manager
            .as_ref()
            .and_then(|manager| manager.auth());

        // Build prompt conversation for MCP sampling request.
        // Per MCP spec: use the provided systemPrompt, or empty string if not provided.
        let system_text = params.system_prompt.clone();
        let items = convert_sampling_messages_to_items(&params)?;
        let mut conversation: Vec<ResponseItem> =
            items.into_iter().map(ResponseItem::from).collect();

        // Select model based on preferences or use config's default
        let model_name = select_model_from_preferences(&params, &config);
        let model_family = find_family_for_model(&model_name).unwrap_or_else(|| {
            let default_family = derive_default_model_family(&model_name);
            debug!(
                "Using default model family for sampling: model={model_name}, family={}",
                default_family.slug
            );
            default_family
        });
        let mut stop_reason: Option<String> = None;

        let session = {
            let guard = self.session.read().await;
            guard.as_ref().and_then(|weak| weak.upgrade())
        };
        let turn_context = if let Some(session) = &session {
            Some(
                session
                    .new_turn_with_sub_id(
                        sampling_sub_id.clone(),
                        SessionSettingsUpdate::default(),
                    )
                    .await,
            )
        } else {
            None
        };
        let tool_specs = session
            .as_ref()
            .map(|session| {
                let tools_config = build_tools_config(&config, &model_family);
                let mcp_tools = session.services.mcp_connection_manager.list_all_tools();
                let router = ToolRouter::from_config(&tools_config, Some(mcp_tools));
                router.specs()
            })
            .unwrap_or_default();
        let parallel_tool_calls =
            model_family.supports_parallel_tool_calls && !tool_specs.is_empty();

        let mut response_text = String::new();
        let mut remaining_iterations = 4;

        loop {
            if remaining_iterations == 0 {
                warn!("Sampling handler exceeded tool iteration limit");
                return Err(rmcp::ErrorData::internal_error(
                    "Sampling request exceeded tool iteration limit",
                    None,
                ));
            }
            remaining_iterations -= 1;

        let prompt = Prompt {
            input: conversation.clone(),
            tools: tool_specs.clone(),
            parallel_tool_calls,
            base_instructions_override: None,
            output_schema: None,
        };

            let response_stream = call_llm_for_sampling(
                &prompt,
                &model_name,
                &config,
                auth_manager.clone(),
                auth.clone(),
            )
            .await?;

            let SamplingTurnResult {
                new_items,
                tool_responses,
                final_text,
                stop_reason: turn_stop_reason,
            } = process_sampling_stream(
                response_stream,
                session.clone(),
                turn_context.clone(),
            )
            .await?;

            if let Some(reason) = turn_stop_reason {
                stop_reason = Some(reason);
            }
            if let Some(text) = final_text {
                response_text = text;
            }

            conversation.extend(new_items.into_iter());

            if tool_responses.is_empty() {
                break;
            }

            // Clear any previous assistant text since we're about to continue the loop with
            // additional tool interactions. This avoids returning stale responses from prior
            // iterations (e.g. an earlier tool failure message) when the model doesn't emit a
            // fresh assistant message after the follow-up tool calls.
            response_text.clear();

            for response in tool_responses {
                conversation.push(ResponseItem::from(response));
            }
        }

        if response_text.trim().is_empty() {
            response_text = extract_last_assistant_message(&conversation)
                .unwrap_or_else(|| "No response from model".to_string());
        }

        info!(
            "Generated sampling response with {} characters",
            response_text.len()
        );

        self.notify_sampling_response(&response_text, &sampling_sub_id)
            .await;

        Ok(CreateMessageResult {
            message: SamplingMessage {
                role: Role::Assistant,
                content: Content::text(&response_text),
            },
            model: model_name,
            stop_reason,
        })
    }
}

/// Select model based on MCP sampling preferences or fall back to config default.
fn select_model_from_preferences(
    params: &rmcp::model::CreateMessageRequestParam,
    config: &Config,
) -> String {
    if let Some(prefs) = &params.model_preferences
        && let Some(hints) = &prefs.hints
    {
        for hint in hints {
            if let Some(name) = &hint.name {
                return name.clone();
            }
        }
    }

    // Fall back to config's model family slug
    config.model_family.slug.clone()
}

/// Call the LLM directly for sampling, bypassing the session's ModelClient.
async fn call_llm_for_sampling(
    prompt: &Prompt,
    model: &str,
    config: &Config,
    auth_manager: Option<Arc<AuthManager>>,
    auth: Option<CodexAuth>,
) -> Result<crate::client_common::ResponseStream, rmcp::ErrorData> {
    use codex_otel::otel_event_manager::OtelEventManager;

    // Try to find a known model family, or use a default one
    let model_family = find_family_for_model(model).unwrap_or_else(|| {
        let default_family = derive_default_model_family(model);
        debug!(
            "Using default model family for sampling: model={model}, family={}",
            default_family.slug
        );
        default_family
    });

    let mut temp_config = config.clone();
    temp_config.model = model.to_string();
    temp_config.model_family = model_family.clone();
    let provider = temp_config.model_provider.clone();
    let temp_config = Arc::new(temp_config);
    let conversation_id = ConversationId::new();
    let (account_id, account_email, auth_mode) = auth
        .as_ref()
        .map(|a| (a.get_account_id(), a.get_account_email(), Some(a.mode)))
        .unwrap_or((None, None, None));
    let otel_manager = OtelEventManager::new(
        conversation_id,
        model,
        &model_family.slug,
        account_id,
        account_email,
        auth_mode,
        false, // Don't log user prompts
        "mcp-sampling".to_string(),
    );

    let model_client = ModelClient::new(
        temp_config.clone(),
        auth_manager.clone(),
        otel_manager,
        provider,
        temp_config.model_reasoning_effort,
        temp_config.model_reasoning_summary,
        conversation_id,
        SessionSource::Mcp,
    );

    model_client.stream(prompt).await.map_err(|err| {
        warn!("LLM call failed for sampling: {err}");
        rmcp::ErrorData::internal_error(format!("LLM call failed: {err}"), None)
    })
}

/// Convert MCP sampling messages to Codex response input items.
fn convert_sampling_messages_to_items(
    params: &rmcp::model::CreateMessageRequestParam,
) -> Result<Vec<ResponseInputItem>, rmcp::ErrorData> {
    use rmcp::model::Role;

    let mut items = Vec::new();

    // Convert each sampling message
    for msg in &params.messages {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .to_string();

        let content = convert_message_content(&msg.content)?;
        items.push(ResponseInputItem::Message { role, content });
    }

    Ok(items)
}

/// Convert MCP message content to Codex content items.
fn convert_message_content(
    content: &rmcp::model::Content,
) -> Result<Vec<ContentItem>, rmcp::ErrorData> {
    match &content.raw {
        rmcp::model::RawContent::Text(text_content) => Ok(vec![ContentItem::InputText {
            text: text_content.text.clone(),
        }]),
        rmcp::model::RawContent::Image(_) => {
            warn!("Image content in sampling messages is not yet supported");
            Err(rmcp::ErrorData::invalid_params(
                "Image content is not supported",
                None,
            ))
        }
        rmcp::model::RawContent::Resource(_) | rmcp::model::RawContent::ResourceLink(_) => {
            warn!("Resource content in sampling messages is not yet supported");
            Err(rmcp::ErrorData::invalid_params(
                "Resource content is not supported",
                None,
            ))
        }
        rmcp::model::RawContent::Audio(_) => {
            warn!("Audio content in sampling messages is not yet supported");
            Err(rmcp::ErrorData::invalid_params(
                "Audio content is not supported",
                None,
            ))
        }
    }
}

struct SamplingTurnResult {
    new_items: Vec<ResponseItem>,
    tool_responses: Vec<ResponseInputItem>,
    final_text: Option<String>,
    stop_reason: Option<String>,
}

async fn process_sampling_stream(
    response_stream: crate::client_common::ResponseStream,
    session: Option<Arc<Session>>,
    turn_context: Option<Arc<TurnContext>>,
) -> Result<SamplingTurnResult, rmcp::ErrorData> {
    use rmcp::model::CreateMessageResult;

    let mut new_items = Vec::new();
    let mut tool_responses = Vec::new();
    let mut final_text = None;
    let mut stop_reason = None;

    tokio::pin!(response_stream);

    while let Some(event_result) = response_stream.next().await {
        match event_result {
            Ok(ResponseEvent::OutputItemDone(item)) => {
                if let ResponseItem::Message { role, content, .. } = &item {
                    if role == "assistant" {
                        if let Some(text) = extract_output_text(content) {
                            final_text = Some(text);
                        }
                    }
                }

                if let ResponseItem::FunctionCall {
                    name,
                    arguments,
                    call_id,
                    ..
                } = &item
                {
                    let response = handle_sampling_function_call(
                        session.as_ref(),
                        turn_context.as_ref(),
                        name,
                        arguments,
                        call_id,
                    )
                    .await;
                    tool_responses.push(response);
                }

                new_items.push(item);
            }
            Ok(ResponseEvent::OutputItemAdded(item)) => {
                if let ResponseItem::Message { role, content, .. } = &item {
                    if role == "assistant" {
                        if let Some(text) = extract_output_text(content) {
                            final_text = Some(text);
                        }
                    }
                }
                // Ignore tool dispatch for added items because arguments may still be streaming.
            }
            Ok(ResponseEvent::OutputTextDelta(_)) => {}
            Ok(ResponseEvent::Completed { .. }) => {
                stop_reason = Some(CreateMessageResult::STOP_REASON_END_TURN.to_string());
            }
            Ok(ResponseEvent::Created)
            | Ok(ResponseEvent::RateLimits(_))
            | Ok(ResponseEvent::ReasoningSummaryDelta(_))
            | Ok(ResponseEvent::ReasoningContentDelta(_))
            | Ok(ResponseEvent::ReasoningSummaryPartAdded) => {}
            Err(err) => {
                warn!("Error in response stream: {err}");
                return Err(rmcp::ErrorData::internal_error(
                    format!("Stream error: {err}"),
                    None,
                ));
            }
        }
    }

    if new_items.is_empty() && final_text.is_none() {
        warn!("Model returned empty response during sampling");
        final_text = Some("No response from model".to_string());
    }

    Ok(SamplingTurnResult {
        new_items,
        tool_responses,
        final_text,
        stop_reason,
    })
}

async fn handle_sampling_function_call(
    session: Option<&Arc<Session>>,
    turn_context: Option<&Arc<TurnContext>>,
    name: &str,
    arguments: &str,
    call_id: &str,
) -> ResponseInputItem {
    if let (Some(session), Some(turn_context)) = (session, turn_context) {
        if let Some((server, tool)) = session.parse_mcp_tool_name(name) {
            return handle_mcp_tool_call(
                session.as_ref(),
                turn_context.as_ref(),
                call_id.to_string(),
                server,
                tool,
                arguments.to_string(),
            )
            .await;
        }
    }

    warn!("Unsupported tool call during sampling: {name}");
    sampling_tool_failure(
        call_id,
        format!("Tool '{name}' is not available during sampling."),
    )
}

fn sampling_tool_failure(call_id: &str, message: impl Into<String>) -> ResponseInputItem {
    let message = message.into();
    ResponseInputItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload {
            content: message,
            content_items: None,
            success: Some(false),
        },
    }
}

fn extract_output_text(content: &[ContentItem]) -> Option<String> {
    let mut aggregated = String::new();
    for piece in content {
        if let ContentItem::OutputText { text } = piece {
            aggregated.push_str(text);
        }
    }
    if aggregated.is_empty() {
        None
    } else {
        Some(aggregated)
    }
}

fn extract_last_assistant_message(items: &[ResponseItem]) -> Option<String> {
    items.iter().rev().find_map(|item| {
        if let ResponseItem::Message { role, content, .. } = item {
            if role == "assistant" {
                return extract_output_text(content);
            }
        }
        None
    })
}

fn build_tools_config(config: &Config, model_family: &crate::model_family::ModelFamily) -> ToolsConfig {
    ToolsConfig::new(&ToolsConfigParams {
        model_family,
        features: &config.features,
    })
}

impl CodexSamplingHandler {
    async fn notify_sampling_request(
        &self,
        params: &rmcp::model::CreateMessageRequestParam,
        sub_id: &str,
    ) {
        let summary = build_sampling_request_summary(params);
        let session = {
            let guard = self.session.read().await;
            guard.as_ref().and_then(|weak| weak.upgrade())
        };

        if let Some(session) = session {
            session
                .send_event_raw(Event {
                    id: sub_id.to_string(),
                    msg: EventMsg::AgentMessage(AgentMessageEvent { message: summary }),
                })
                .await;
        } else {
            debug!("Sampling request received, but no active session to notify");
        }
    }

    async fn notify_sampling_response(&self, response_text: &str, sub_id: &str) {
        if let Some(summary) = build_sampling_response_summary(response_text) {
            let session = {
                let guard = self.session.read().await;
                guard.as_ref().and_then(|weak| weak.upgrade())
            };

            if let Some(session) = session {
                session
                    .send_event_raw(Event {
                        id: sub_id.to_string(),
                        msg: EventMsg::AgentMessage(AgentMessageEvent { message: summary }),
                    })
                    .await;
            } else {
                debug!("Sampling response ready, but no active session to notify");
            }
        }
    }
}

fn build_sampling_request_summary(
    params: &rmcp::model::CreateMessageRequestParam,
) -> String {
    use rmcp::model::RawContent;
    use rmcp::model::Role;

    let mut message = String::new();
    let _ = writeln!(&mut message, "MCP sampling request received.");
    let _ = writeln!(
        &mut message,
        "- messages: {}",
        params.messages.len()
    );

    if params.max_tokens > 0 {
        let max_tokens = params.max_tokens;
        let _ = writeln!(&mut message, "- max tokens: {max_tokens}");
    }
    if let Some(temperature) = params.temperature {
        let _ = writeln!(&mut message, "- temperature: {temperature}");
    }
    if let Some(prefs) = &params.model_preferences {
        if let Ok(serialized) = serde_json::to_string(prefs) {
            let _ = writeln!(
                &mut message,
                "- model preferences: {}",
                truncate_snippet(&serialized, 200)
            );
        }
    }
    if let Some(system_prompt) = params
        .system_prompt
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let _ = writeln!(
            &mut message,
            "- system prompt: \"{}\"",
            truncate_snippet(system_prompt, 200)
        );
    }

    for (idx, msg) in params.messages.iter().enumerate() {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        let snippet = match &msg.content.raw {
            RawContent::Text(text) => truncate_snippet(&text.text, 200),
            RawContent::Resource(_) | RawContent::ResourceLink(_) => "[resource content]".into(),
            RawContent::Image(_) => "[image content]".into(),
            RawContent::Audio(_) => "[audio content]".into(),
        };

        let _ = writeln!(&mut message, "  - #{idx} {role}: {snippet}");
        if idx >= 4 {
            let _ = writeln!(&mut message, "  - ...");
            break;
        }
    }

    message.trim_end().to_string()
}

fn truncate_snippet(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut truncated = text[..max_len].to_string();
    truncated.push_str("...");
    truncated
}

fn build_sampling_response_summary(response_text: &str) -> Option<String> {
    use serde_json::Value;

    let trimmed = response_text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let json: Value = serde_json::from_str(trimmed).ok()?;

    let mut message = String::new();
    let _ = writeln!(&mut message, "Sampling response received.");

    if let Some(decisions) = json.get("decisions").and_then(|d| d.as_array()) {
        if !decisions.is_empty() {
            let _ = writeln!(&mut message, "- tool actions:");
            for decision in decisions.iter().take(4) {
                let action = decision
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let status = decision
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let note = decision
                    .get("note")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty());

                let _ = if let Some(note) = note {
                    writeln!(
                        &mut message,
                        "  - {action} (status: {status}, note: {})",
                        truncate_snippet(note, 12000)
                    )
                } else {
                    writeln!(&mut message, "  - {action} (status: {status})")
                };
            }

            if decisions.len() > 4 {
                let remaining = decisions.len() - 4;
                let _ = writeln!(&mut message, "  - ... {remaining} more decision(s)");
            }
        }
    }

    let _ = writeln!(
        &mut message,
        "- raw response: {}",
        truncate_snippet(trimmed, 3000)
    );

    Some(message.trim_end().to_string())
}
