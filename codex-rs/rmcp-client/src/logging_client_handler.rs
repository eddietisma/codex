use rmcp::ClientHandler;
use rmcp::RoleClient;
use rmcp::model::CancelledNotificationParam;
use rmcp::model::ClientInfo;
use rmcp::model::CreateElicitationRequestParam;
use rmcp::model::CreateElicitationResult;
use rmcp::model::CreateMessageRequestParam;
use rmcp::model::CreateMessageResult;
use rmcp::model::ElicitationAction;
use rmcp::model::LoggingLevel;
use rmcp::model::LoggingMessageNotificationParam;
use rmcp::model::ProgressNotificationParam;
use rmcp::model::ResourceUpdatedNotificationParam;
use rmcp::service::NotificationContext;
use rmcp::service::RequestContext;
use std::sync::Arc;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::SamplingHandler;

#[derive(Clone)]
pub(crate) struct LoggingClientHandler {
    client_info: ClientInfo,
    sampling_handler: Option<Arc<dyn SamplingHandler>>,
}

impl std::fmt::Debug for LoggingClientHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoggingClientHandler")
            .field("client_info", &self.client_info)
            .field(
                "sampling_handler",
                &self.sampling_handler.as_ref().map(|_| "Some(...)"),
            )
            .finish()
    }
}

impl LoggingClientHandler {
    pub(crate) fn new(client_info: ClientInfo) -> Self {
        Self {
            client_info,
            sampling_handler: None,
        }
    }

    pub(crate) fn with_sampling_handler(
        client_info: ClientInfo,
        handler: Arc<dyn SamplingHandler>,
    ) -> Self {
        Self {
            client_info,
            sampling_handler: Some(handler),
        }
    }
}

impl ClientHandler for LoggingClientHandler {
    // TODO (CODEX-3571): support elicitations.
    async fn create_elicitation(
        &self,
        request: CreateElicitationRequestParam,
        _context: RequestContext<RoleClient>,
    ) -> Result<CreateElicitationResult, rmcp::ErrorData> {
        info!(
            "MCP server requested elicitation ({}). Elicitations are not supported yet. Declining.",
            request.message
        );
        Ok(CreateElicitationResult {
            action: ElicitationAction::Decline,
            content: None,
        })
    }

    async fn create_message(
        &self,
        params: CreateMessageRequestParam,
        _context: RequestContext<RoleClient>,
    ) -> Result<CreateMessageResult, rmcp::ErrorData> {
        info!(
            "MCP server requested sampling with {} messages",
            params.messages.len()
        );

        if let Some(handler) = &self.sampling_handler {
            handler.create_message(params).await
        } else {
            warn!("Sampling request received but sampling is not configured");
            Err(rmcp::ErrorData::internal_error(
                "Sampling is not enabled for this client",
                None,
            ))
        }
    }

    async fn on_cancelled(
        &self,
        params: CancelledNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!(
            "MCP server cancelled request (request_id: {}, reason: {:?})",
            params.request_id, params.reason
        );
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!(
            "MCP server progress notification (token: {:?}, progress: {}, total: {:?}, message: {:?})",
            params.progress_token, params.progress, params.total, params.message
        );
    }

    async fn on_resource_updated(
        &self,
        params: ResourceUpdatedNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        info!("MCP server resource updated (uri: {})", params.uri);
    }

    async fn on_resource_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!("MCP server resource list changed");
    }

    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!("MCP server tool list changed");
    }

    async fn on_prompt_list_changed(&self, _context: NotificationContext<RoleClient>) {
        info!("MCP server prompt list changed");
    }

    fn get_info(&self) -> ClientInfo {
        self.client_info.clone()
    }

    async fn on_logging_message(
        &self,
        params: LoggingMessageNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        let LoggingMessageNotificationParam {
            level,
            logger,
            data,
        } = params;
        let logger = logger.as_deref();
        match level {
            LoggingLevel::Emergency
            | LoggingLevel::Alert
            | LoggingLevel::Critical
            | LoggingLevel::Error => {
                error!(
                    "MCP server log message (level: {:?}, logger: {:?}, data: {})",
                    level, logger, data
                );
            }
            LoggingLevel::Warning => {
                warn!(
                    "MCP server log message (level: {:?}, logger: {:?}, data: {})",
                    level, logger, data
                );
            }
            LoggingLevel::Notice | LoggingLevel::Info => {
                info!(
                    "MCP server log message (level: {:?}, logger: {:?}, data: {})",
                    level, logger, data
                );
            }
            LoggingLevel::Debug => {
                debug!(
                    "MCP server log message (level: {:?}, logger: {:?}, data: {})",
                    level, logger, data
                );
            }
        }
    }
}
