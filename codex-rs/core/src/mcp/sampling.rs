use std::fmt::Write;
use std::sync::{Arc, Weak};
use std::time::Duration;

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
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::ToolRouter;
use crate::tools::spec::ToolsConfig;
use crate::tools::spec::ToolsConfigParams;
use crate::turn_diff_tracker::TurnDiffTracker;
use crate::AuthManager;
use codex_protocol::ConversationId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::McpInvocation;
use codex_protocol::protocol::McpToolCallBeginEvent;
use codex_protocol::protocol::McpToolCallEndEvent;
use codex_protocol::protocol::SessionSource;
use mcp_types::CallToolResult;
use mcp_types::ContentBlock;
use mcp_types::TextContent;
use serde_json::json;

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
        let (tool_router, tool_specs) = session
            .as_ref()
            .map(|session| {
                let tools_config = build_tools_config(&config, &model_family);
                let mcp_tools = session.services.mcp_connection_manager.list_all_tools();
                let router = ToolRouter::from_config(&tools_config, Some(mcp_tools));
                let specs = router.specs();
                (Some(Arc::new(router)), specs)
            })
            .unwrap_or((None, Vec::new()));
        let parallel_tool_calls =
            model_family.supports_parallel_tool_calls && !tool_specs.is_empty();

        let mut response_text = String::new();
        let mut remaining_iterations = 999999; // Use a large limit for long-running sampling

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
                tool_router.clone(),
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
    tool_router: Option<Arc<ToolRouter>>,
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

                match &item {
                    ResponseItem::FunctionCall {
                        name,
                        arguments,
                        call_id,
                        ..
                    } => {
                        let response = handle_sampling_function_call(
                            session.as_ref(),
                            turn_context.as_ref(),
                            tool_router.as_ref(),
                            name,
                            arguments,
                            call_id,
                        )
                        .await;
                        tool_responses.push(response);
                    }
                    ResponseItem::CustomToolCall { name, input, call_id, .. } => {
                        let response = handle_sampling_custom_tool_call(
                            session.as_ref(),
                            turn_context.as_ref(),
                            tool_router.as_ref(),
                            name,
                            input,
                            call_id,
                        )
                        .await;
                        tool_responses.push(response);
                    }
                    _ => {}
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
    tool_router: Option<&Arc<ToolRouter>>,
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

        if let Some(router) = tool_router {
            let item = ResponseItem::FunctionCall {
                id: None,
                name: name.to_string(),
                arguments: arguments.to_string(),
                call_id: call_id.to_string(),
            };

            match ToolRouter::build_tool_call(session.as_ref(), item) {
                Ok(Some(call)) => {
                    let tracker: SharedTurnDiffTracker =
                        Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new()));
                    return match router
                        .dispatch_tool_call(
                            Arc::clone(session),
                            Arc::clone(turn_context),
                            tracker,
                            call,
                        )
                        .await
                    {
                        Ok(response) => response,
                        Err(err) => {
                            warn!(
                                "Tool dispatch error during sampling for `{name}`: {err}"
                            );
                            sampling_tool_failure(
                                call_id,
                                format!("Tool '{name}' failed during sampling: {err}"),
                            )
                        }
                    };
                }
                Ok(None) => {
                    warn!(
                        "Function call `{name}` did not produce a tool invocation during sampling"
                    );
                    return sampling_tool_failure(
                        call_id,
                        format!("Tool '{name}' is not available during sampling."),
                    );
                }
                Err(err) => {
                    warn!(
                        "Failed to build tool call for `{name}` during sampling: {err}"
                    );
                    return sampling_tool_failure(
                        call_id,
                        format!("Tool '{name}' is not available during sampling: {err}"),
                    );
                }
            }
        }
    }

    warn!("Unsupported tool call during sampling: {name}");
    sampling_tool_failure(
        call_id,
        format!("Tool '{name}' is not available during sampling."),
    )
}

async fn handle_sampling_custom_tool_call(
    session: Option<&Arc<Session>>,
    turn_context: Option<&Arc<TurnContext>>,
    tool_router: Option<&Arc<ToolRouter>>,
    name: &str,
    input: &str,
    call_id: &str,
) -> ResponseInputItem {
    if let (Some(session), Some(turn_context), Some(router)) =
        (session, turn_context, tool_router)
    {
        let item = ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id: call_id.to_string(),
            name: name.to_string(),
            input: input.to_string(),
        };

        match ToolRouter::build_tool_call(session.as_ref(), item) {
            Ok(Some(call)) => {
                let tracker: SharedTurnDiffTracker =
                    Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new()));
                return match router
                    .dispatch_tool_call(
                        Arc::clone(session),
                        Arc::clone(turn_context),
                        tracker,
                        call,
                    )
                    .await
                {
                    Ok(response) => response,
                    Err(err) => {
                        warn!(
                            "Custom tool dispatch error during sampling for `{name}`: {err}"
                        );
                        sampling_custom_tool_failure(
                            call_id,
                            format!("Tool '{name}' failed during sampling: {err}"),
                        )
                    }
                };
            }
            Ok(None) => {
                warn!(
                    "Custom tool `{name}` did not produce a tool invocation during sampling"
                );
                return sampling_custom_tool_failure(
                    call_id,
                    format!("Tool '{name}' is not available during sampling."),
                );
            }
            Err(err) => {
                warn!(
                    "Failed to build custom tool call for `{name}` during sampling: {err}"
                );
                return sampling_custom_tool_failure(
                    call_id,
                    format!("Tool '{name}' is not available during sampling: {err}"),
                );
            }
        }
    }

    warn!("Unsupported custom tool call during sampling: {name}");
    sampling_custom_tool_failure(
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

fn sampling_custom_tool_failure(
    call_id: &str,
    message: impl Into<String>,
) -> ResponseInputItem {
    ResponseInputItem::CustomToolCallOutput {
        call_id: call_id.to_string(),
        output: message.into(),
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
        let Some(summary) = build_sampling_response_summary(response_text) else {
            return;
        };

        let session = {
            let guard = self.session.read().await;
            guard.as_ref().and_then(|weak| weak.upgrade())
        };

        let Some(session) = session else {
            debug!("Sampling response ready, but no active session to notify");
            return;
        };

        let call_id = format!("{sub_id}-summary");
        let invocation = McpInvocation {
            server: "codex-sampling".to_string(),
            tool: "response_summary".to_string(),
            arguments: Some(json!({
                "source": "sampling_handler",
                "preview": truncate_snippet(&summary, 200),
            })),
        };

        session
            .send_event_raw(Event {
                id: sub_id.to_string(),
                msg: EventMsg::McpToolCallBegin(McpToolCallBeginEvent {
                    call_id: call_id.clone(),
                    invocation: invocation.clone(),
                }),
            })
            .await;

        let result = CallToolResult {
            content: vec![ContentBlock::TextContent(TextContent {
                annotations: None,
                text: summary,
                r#type: "text".to_string(),
            })],
            is_error: Some(false),
            structured_content: None,
        };

        session
            .send_event_raw(Event {
                id: sub_id.to_string(),
                msg: EventMsg::McpToolCallEnd(McpToolCallEndEvent {
                    call_id,
                    invocation,
                    duration: Duration::from_millis(0),
                    result: Ok(result),
                }),
            })
            .await;
    }
}

fn build_sampling_request_summary(
    params: &rmcp::model::CreateMessageRequestParam,
) -> String {
    use rmcp::model::RawContent;
    use rmcp::model::Role;

    const HEADER: &str = "\x1b[36m";
    const DIM: &str = "\x1b[90m";
    const WHITE: &str = "\x1b[97m";
    const RESET: &str = "\x1b[0m";
    const BRANCH: &str = "└─ ";
    const SUB_BRANCH: &str = "   └─ ";
    const CONTENT_INDENT: &str = "      ";

    let mut message = String::new();
    let max_line_len = determine_sampling_line_limit();
    let mut meta_fields = vec![format!("\"messages\":{}", params.messages.len())];
    if params.max_tokens > 0 {
        meta_fields.push(format!("\"maxTokens\":{}", params.max_tokens));
    }
    if let Some(temperature) = params.temperature {
        meta_fields.push(format!("\"temperature\":{}", temperature));
    }
    if let Some(prefs) = &params.model_preferences {
        if let Ok(serialized) = serde_json::to_string(prefs) {
            meta_fields.push(format!(
                "\"preferences\":{}",
                truncate_snippet(&serialized, max_line_len.saturating_sub(6))
            ));
        }
    }
    let meta_joined = meta_fields.join(",");
    let meta_display = truncate_snippet(&meta_joined, max_line_len.saturating_sub(6));
    let _ = writeln!(
        &mut message,
        "{WHITE}Received {HEADER}sampling{RESET}{WHITE}({DIM}{{{meta_display}}}{WHITE}){RESET}"
    );

    if let Some(system_prompt) = params
        .system_prompt
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        let _ = writeln!(&mut message, "{DIM}{BRANCH}system prompt:{RESET}");
        for line in system_prompt.lines() {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                let _ = writeln!(&mut message, "{DIM}{CONTENT_INDENT}{RESET}");
            } else {
                let _ = writeln!(
                    &mut message,
                    "{DIM}{CONTENT_INDENT}{}{RESET}",
                    truncate_snippet(trimmed, max_line_len)
                );
            }
        }
    }

    if !params.messages.is_empty() {
        let _ = writeln!(
            &mut message,
            "{DIM}{BRANCH}messages:{RESET}"
        );
    }

    for (idx, msg) in params.messages.iter().enumerate() {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        let snippet = match &msg.content.raw {
            RawContent::Text(text) => text.text.clone(),
            RawContent::Resource(_) | RawContent::ResourceLink(_) => "[resource content]".into(),
            RawContent::Image(_) => "[image content]".into(),
            RawContent::Audio(_) => "[audio content]".into(),
        };

        let _ = writeln!(
            &mut message,
            "{DIM}{SUB_BRANCH}#{idx} ({role}):{RESET}"
        );
        for line in snippet.lines() {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                let _ = writeln!(&mut message, "{DIM}{CONTENT_INDENT}{RESET}");
            } else {
                let _ = writeln!(
                    &mut message,
                    "{DIM}{CONTENT_INDENT}{}{RESET}",
                    truncate_snippet(trimmed, max_line_len)
                );
            }
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

fn determine_sampling_line_limit() -> usize {
    const DEFAULT_MAX: usize = 160;
    const MIN: usize = 40;
    const PADDING: usize = 10;

    let width = textwrap::termwidth();
    if width == 0 {
        return DEFAULT_MAX;
    }

    width
        .saturating_sub(PADDING)
        .clamp(MIN, DEFAULT_MAX)
}

fn build_sampling_response_summary(response_text: &str) -> Option<String> {
    use serde_json::Value;

    let trimmed = response_text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let json: Value = serde_json::from_str(trimmed).ok()?;

    let mut message = String::new();
    if let Some(decisions) = json.get("decisions").and_then(|d| d.as_array()) {
        if !decisions.is_empty() {
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

                match note {
                    Some(note) => {
                        let _ = writeln!(
                            &mut message,
                            "{action} — {status} ({})",
                            truncate_snippet(note, 12000)
                        );
                    }
                    None => {
                        let _ = writeln!(&mut message, "{action} — {status}");
                    }
                }
            }

            if decisions.len() > 4 {
                let remaining = decisions.len() - 4;
                let _ = writeln!(&mut message, "... {remaining} more decision(s)");
            }
        }
    }

    if message.is_empty() {
        return None;
    }

    Some(message.trim_end().to_string())
}
