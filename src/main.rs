slint::include_modules!();
use futures::StreamExt;
use ollama_rs::Ollama;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use ollama_rs::generation::completion::request::GenerationRequest;
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

struct AppState {
    db: Connection,
    current_session_id: String,
    chat_history: Vec<ChatMessage>,
}

#[tokio::main]
async fn main() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;
    let ollama = Ollama::default();

    // Database Initialization
    let db = Connection::open("history.db").expect("Failed to open DB");
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)", []).unwrap();
    db.execute("CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT)", []).unwrap();

    let state = Arc::new(Mutex::new(AppState {
        db,
        current_session_id: Uuid::new_v4().to_string(),
        chat_history: Vec::new(),
    }));

    let ui_handle = ui.as_weak();
    refresh_history(&ui_handle, &state.lock().unwrap().db);

    // Load Persistent Settings
    let cfg: serde_json::Value = confy::load("ollama-native", None).unwrap_or(serde_json::json!({
        "default_model": "llama3",
        "scroll_lock": true
    }));
    ui.set_default_model_setting(cfg["default_model"].as_str().unwrap_or("llama3").into());
    ui.set_selected_model(cfg["default_model"].as_str().unwrap_or("llama3").into());
    ui.set_scroll_lock(cfg["scroll_lock"].as_bool().unwrap_or(true));

    // Populate Model List
    let o_models = ollama.clone();
    let u_models = ui_handle.clone();
    tokio::spawn(async move {
        if let Ok(models) = o_models.list_local_models().await {
            let names: Vec<slint::SharedString> = models.into_iter().map(|m| m.name.into()).collect();
            let _ = u_models.upgrade_in_event_loop(move |ui| {
                ui.set_model_list(std::rc::Rc::new(slint::VecModel::from(names)).into());
            });
        }
    });

    // --- Title Generator Helper ---
    let o_title = ollama.clone();
    let finalize_session = move |old_id: String, history: Vec<ChatMessage>| {
        let ollama_c = o_title.clone();
        tokio::spawn(async move {
            if history.is_empty() { return; }
            let mut prompt = String::from("Summarise this chat session into a short title of max 7 words. Respond ONLY with the title. No quotes.\n\nContent:\n");
            for m in history.iter().take(4) { // Use first few messages for context
                let role = if m.role == ollama_rs::generation::chat::MessageRole::User { "User" } else { "AI" };
                prompt.push_str(&format!("{}: {}\n", role, m.content));
            }
            if let Ok(res) = ollama_c.generate(GenerationRequest::new("llama3".into(), prompt)).await {
                if let Ok(db_conn) = Connection::open("history.db") {
                    let _ = db_conn.execute("UPDATE sessions SET title = ?1 WHERE id = ?2", params![res.response.trim(), old_id]);
                }
            }
        });
    };

    // --- UI Callbacks ---

    ui.on_set_default_model(move |model| {
        let _ = confy::store("ollama-native", None, serde_json::json!({ "default_model": model.to_string() }));
    });

    let s_load = state.clone();
    let u_load = ui_handle.clone();
    let f_load = finalize_session.clone();
    ui.on_load_session(move |id| {
        let mut s = s_load.lock().unwrap();
        // Save title for current session before switching
        f_load(s.current_session_id.clone(), s.chat_history.clone());

        let id_str = id.to_string();
        let mut history_to_load = Vec::new();
        {
            let mut stmt = s.db.prepare("SELECT role, content FROM messages WHERE session_id = ?1").unwrap();
            let rows = stmt.query_map([&id_str], |row| {
                let role: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok(if role == "user" { ChatMessage::user(content) } else { ChatMessage::assistant(content) })
            }).unwrap();
            for r in rows { history_to_load.push(r.unwrap()); }
        }
        s.chat_history = history_to_load;
        s.current_session_id = id_str;
        update_ui_text(&u_load, &s.chat_history);
    });

    let s_clear = state.clone();
    let u_clear = ui_handle.clone();
    let f_clear = finalize_session.clone();
    ui.on_clear_chat(move || {
        let mut s = s_clear.lock().unwrap();
        f_clear(s.current_session_id.clone(), s.chat_history.clone());
        s.current_session_id = Uuid::new_v4().to_string();
        s.chat_history.clear();
        let _ = u_clear.upgrade_in_event_loop(|ui| ui.set_chat_text("".into()));
        refresh_history(&u_clear, &s.db);
    });

    let s_send = state.clone();
    let u_send = ui_handle.clone();
    let o_send = ollama.clone();
    ui.on_send_message(move |msg| {
        let mut s = s_send.lock().unwrap();
        let msg_str = msg.to_string();
        let session_id = s.current_session_id.clone();

        // Create session in DB if this is the first message
        if s.chat_history.is_empty() {
            let _ = s.db.execute("INSERT INTO sessions (id, title, created_at) VALUES (?1, ?2, datetime('now'))", params![session_id, msg_str]);
        }

        s.chat_history.push(ChatMessage::user(msg_str.clone()));
        let _ = s.db.execute("INSERT INTO messages (session_id, role, content) VALUES (?1, 'user', ?2)", params![session_id, msg_str]);
        update_ui_text(&u_send, &s.chat_history);

        let model = u_send.upgrade().map(|ui| ui.get_selected_model().to_string()).unwrap_or_else(|| "llama3".into());
        let o_client = o_send.clone();
        let history = s.chat_history.clone();
        let inner_u = u_send.clone();
        let inner_s = s_send.clone();

        tokio::spawn(async move {
            let req = ChatMessageRequest::new(model.clone(), history);
            if let Ok(mut stream) = o_client.send_chat_messages_stream(req).await {
                let mut full_response = String::new();
                let mut first_chunk = true;
                while let Some(Ok(res)) = stream.next().await {
                    let chunk = res.message.content;
                    full_response.push_str(&chunk);
                    let ui_copy = inner_u.clone();
                    let chunk_val = chunk.clone();
                    let model_val = model.clone();
                    let is_first = first_chunk;
                    first_chunk = false;

                    let _ = ui_copy.upgrade_in_event_loop(move |ui| {
                        let mut current = ui.get_chat_text().to_string();
                        if is_first { current.push_str(&format!("\n\n{}: ", model_val)); }
                        current.push_str(&chunk_val);
                        ui.set_chat_text(current.into());
                    });
                }
                let mut s_final = inner_s.lock().unwrap();
                s_final.chat_history.push(ChatMessage::assistant(full_response.clone()));
                let _ = s_final.db.execute("INSERT INTO messages (session_id, role, content) VALUES (?1, 'assistant', ?2)", params![session_id, full_response]);
                refresh_history(&inner_u, &s_final.db);
            }
        });
    });

    ui.run()
}

fn update_ui_text(ui: &slint::Weak<AppWindow>, history: &[ChatMessage]) {
    let text = history.iter()
        .map(|m| format!("{}: {}", if m.role == ollama_rs::generation::chat::MessageRole::User { "User" } else { "AI" }, m.content))
        .collect::<Vec<_>>().join("\n\n");
    let _ = ui.upgrade_in_event_loop(move |ui| ui.set_chat_text(text.into()));
}

fn refresh_history(ui: &slint::Weak<AppWindow>, db: &Connection) {
    let mut stmt = db.prepare("SELECT id, title FROM sessions ORDER BY created_at DESC").unwrap();
    let history_items: Vec<HistoryEntry> = stmt.query_map([], |row| {
        Ok(HistoryEntry {
            id: row.get::<_, String>(0).unwrap().into(),
            title: row.get::<_, String>(1).unwrap().into()
        })
    }).unwrap().map(|r| r.unwrap()).collect();

    let _ = ui.upgrade_in_event_loop(move |ui| {
        ui.set_history_list(std::rc::Rc::new(slint::VecModel::from(history_items)).into());
    });
}
