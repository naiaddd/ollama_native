slint::include_modules!();
use futures::StreamExt;
use ollama_rs::Ollama;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use slint::{Model, VecModel, SharedString, ComponentHandle};
use std::rc::Rc;

struct AppState {
    db: Connection,
    current_session_id: String,
    chat_history: Vec<ChatMessage>,
    attachments: Vec<(String, String)>,
}

#[tokio::main]
async fn main() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    let ollama = Ollama::default();

    let db = Connection::open("history.db").expect("Failed to open DB");
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)", []).unwrap();
    db.execute("CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT)", []).unwrap();

    let state = Arc::new(Mutex::new(AppState {
        db,
        current_session_id: Uuid::new_v4().to_string(),
        chat_history: Vec::new(),
        attachments: Vec::new(),
    }));

    let ui_handle = ui.as_weak();
    refresh_history(&ui_handle, &state.lock().unwrap().db);

    let cfg: serde_json::Value = confy::load("ollama-native", None).unwrap_or(serde_json::json!({
        "default_model": "llama3",
        "scroll_lock": true
    }));

    ui.set_default_model_setting(cfg["default_model"].as_str().unwrap_or("llama3").into());
    ui.set_selected_model(cfg["default_model"].as_str().unwrap_or("llama3").into());
    ui.set_scroll_lock(cfg["scroll_lock"].as_bool().unwrap_or(true));

    let o_models = ollama.clone();
    let u_models = ui_handle.clone();
    tokio::spawn(async move {
        if let Ok(models) = o_models.list_local_models().await {
            let names: Vec<SharedString> = models.into_iter().map(|m| m.name.into()).collect();
            let _ = u_models.upgrade_in_event_loop(move |ui| {
                ui.set_model_list(Rc::new(VecModel::from(names)).into());
            });
        }
    });

    let s_pick = state.clone();
    let u_pick = ui_handle.clone();
    ui.on_pick_attachment(move || {
        if let Some(path) = rfd::FileDialog::new().pick_file() {
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(content) = String::from_utf8(bytes) {
                    let mut s = s_pick.lock().unwrap();
                    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    s.attachments.push((filename, content));
                    let names: Vec<SharedString> = s.attachments.iter().map(|(n, _)| n.into()).collect();
                    let _ = u_pick.upgrade_in_event_loop(move |ui| {
                        ui.set_attachment_list(Rc::new(VecModel::from(names)).into());
                    });
                }
            }
        }
    });

    let s_remove = state.clone();
    let u_remove = ui_handle.clone();
    ui.on_remove_attachment(move |index| {
        let mut s = s_remove.lock().unwrap();
        if index >= 0 && (index as usize) < s.attachments.len() {
            s.attachments.remove(index as usize);
            let names: Vec<SharedString> = s.attachments.iter().map(|(n, _)| n.into()).collect();
            let _ = u_remove.upgrade_in_event_loop(move |ui| {
                ui.set_attachment_list(Rc::new(VecModel::from(names)).into());
            });
        }
    });

    let s_load = state.clone();
    let u_load = ui_handle.clone();
    ui.on_load_session(move |id| {
        let mut s = s_load.lock().unwrap();
        let id_str = id.to_string();
        let mut history_to_load = Vec::new();

        // SCOPE FIX: We use a block to ensure stmt is dropped before we mutate s
        {
            let mut stmt = s.db.prepare("SELECT role, content FROM messages WHERE session_id = ?1").unwrap();
            let rows = stmt.query_map([&id_str], |row| {
                let role: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok(if role == "user" { ChatMessage::user(content) } else { ChatMessage::assistant(content) })
            }).unwrap();

            for r in rows {
                if let Ok(msg) = r { history_to_load.push(msg); }
            }
        }

        s.chat_history = history_to_load;
        s.current_session_id = id_str;
        s.attachments.clear();

        let history_copy = s.chat_history.clone();
        let _ = u_load.upgrade_in_event_loop(move |ui| {
            ui.set_attachment_list(Rc::new(VecModel::from(vec![])).into());
            update_ui_model(&ui, &history_copy);
        });
    });

    let s_clear = state.clone();
    let u_clear = ui_handle.clone();
    ui.on_clear_chat(move || {
        let mut s = s_clear.lock().unwrap();
        s.current_session_id = Uuid::new_v4().to_string();
        s.chat_history.clear();
        s.attachments.clear();
        let _ = u_clear.upgrade_in_event_loop(|ui| {
            ui.set_chat_messages(Rc::new(VecModel::from(vec![])).into());
            ui.set_attachment_list(Rc::new(VecModel::from(vec![])).into());
        });
        refresh_history(&u_clear, &s.db);
    });

    let s_send = state.clone();
    let u_send = ui_handle.clone();
    let o_send = ollama.clone();
    ui.on_send_message(move |msg| {
        let mut s = s_send.lock().unwrap();
        let raw_input = msg.to_string();
        let session_id = s.current_session_id.clone();

        let mut full_message = String::new();
        if !s.attachments.is_empty() {
            full_message.push_str("Context from files:\n");
            for (name, content) in &s.attachments {
                full_message.push_str(&format!("[{}]\n{}\n", name, content));
            }
        }
        full_message.push_str(&raw_input);
        s.attachments.clear();

        if s.chat_history.is_empty() {
            let _ = s.db.execute(
                "INSERT INTO sessions (id, title, created_at) VALUES (?1, ?2, datetime('now'))",
                params![session_id, raw_input]
            );
        }

        s.chat_history.push(ChatMessage::user(full_message.clone()));
        let _ = s.db.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?1, 'user', ?2)",
            params![session_id, full_message]
        );

        let model_name = u_send.upgrade().map(|ui| ui.get_selected_model().to_string()).unwrap_or_else(|| "llama3".into());
        let o_client = o_send.clone();

        // MOVE FIX: Clone once for the UI update and once for the tokio thread
        let history_for_ui = s.chat_history.clone();
        let history_for_ai = s.chat_history.clone();

        let inner_u = u_send.clone();
        let inner_s = s_send.clone();

        let _ = inner_u.upgrade_in_event_loop(move |ui| {
            update_ui_model(&ui, &history_for_ui);
        });

        tokio::spawn(async move {
            let req = ChatMessageRequest::new(model_name, history_for_ai);
            if let Ok(mut stream) = o_client.send_chat_messages_stream(req).await {
                let mut full_response = String::new();

                let _ = inner_u.upgrade_in_event_loop(|ui| {
                    let model = ui.get_chat_messages();
                    if let Some(vec_model) = model.as_any().downcast_ref::<VecModel<ChatMessageData>>() {
                        vec_model.push(ChatMessageData { role: "AI".into(), content: "".into() });
                    }
                });

                while let Some(Ok(res)) = stream.next().await {
                    let chunk = res.message.content;
                    full_response.push_str(&chunk);
                    let current_text: SharedString = full_response.clone().into();

                    let _ = inner_u.upgrade_in_event_loop(move |ui| {
                        let model = ui.get_chat_messages();
                        if let Some(vec_model) = model.as_any().downcast_ref::<VecModel<ChatMessageData>>() {
                            let row_idx = vec_model.row_count() - 1;
                            vec_model.set_row_data(row_idx, ChatMessageData {
                                role: "AI".into(),
                                content: current_text
                            });
                        }
                    });
                }

                let mut s_final = inner_s.lock().unwrap();
                s_final.chat_history.push(ChatMessage::assistant(full_response.clone()));
                let _ = s_final.db.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?1, 'assistant', ?2)",
                    params![session_id, full_response]
                );
                refresh_history(&inner_u, &s_final.db);
            }
        });
    });

    ui.run()
}

fn update_ui_model(ui: &AppWindow, history: &[ChatMessage]) {
    let ui_messages: Vec<ChatMessageData> = history.iter()
        .map(|m| ChatMessageData {
            role: if m.role == ollama_rs::generation::chat::MessageRole::User { "User".into() } else { "AI".into() },
            content: m.content.clone().into(),
        })
        .collect();
    ui.set_chat_messages(Rc::new(VecModel::from(ui_messages)).into());
}

fn refresh_history(ui_weak: &slint::Weak<AppWindow>, db: &Connection) {
    let mut stmt = db.prepare("SELECT id, title FROM sessions ORDER BY created_at DESC").unwrap();
    let history_items: Vec<HistoryEntry> = stmt.query_map([], |row| {
        Ok(HistoryEntry {
            id: row.get::<usize, String>(0).unwrap().into(),
            title: row.get::<usize, String>(1).unwrap().into()
        })
    }).unwrap().map(|r| r.unwrap()).collect();

    let _ = ui_weak.upgrade_in_event_loop(move |ui| {
        ui.set_history_list(Rc::new(VecModel::from(history_items)).into());
    });
}
