/* Persistent assistant UI; conversation and message text stays inert. */
(() => {
    const list = document.getElementById("conversation-list");
    if (!list) return;
    const messages = document.getElementById("chat-messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const status = document.getElementById("conversation-list-status");
    const error = document.getElementById("chat-error");
    const mode = document.getElementById("chat-mode");
    let activeThreadId = null;

    function appendTurn(role, content) {
        messages.querySelector(".chat-empty")?.remove();
        const turn = document.createElement("div");
        turn.className = `chat-turn ${role}`;
        const avatar = document.createElement("span");
        avatar.className = "chat-avatar";
        avatar.textContent = role === "user" ? "YOU" : "AI";
        const body = document.createElement("div");
        body.className = "chat-turn-content";
        const label = document.createElement("span");
        label.className = "chat-role";
        label.textContent = role === "user" ? "YOU" : "ASSISTANT";
        const text = document.createElement("span");
        text.textContent = content;
        body.append(label, text);
        turn.append(avatar, body);
        messages.append(turn);
        messages.scrollTop = messages.scrollHeight;
    }

    function renderHistory(turns) {
        messages.replaceChildren();
        if (!turns.length) {
            const empty = document.createElement("div");
            empty.className = "chat-empty";
            const title = document.createElement("h2");
            title.textContent = "Start with a report question";
            const hint = document.createElement("p");
            hint.textContent = "Questions about model outputs are answered from this thread’s linked analysis reports.";
            empty.append(title, hint);
            messages.append(empty);
            return;
        }
        turns.forEach((turn) => appendTurn(turn.role, turn.content));
    }

    async function loadConversation(threadId) {
        const response = await fetch(`/api/conversations/${encodeURIComponent(threadId)}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Could not load conversation.");
        activeThreadId = threadId;
        document.getElementById("chat-thread-label").textContent = threadId;
        renderHistory(payload.messages || []);
        list.querySelectorAll(".thread-button").forEach((button) => {
            button.classList.toggle("active", button.dataset.threadId === threadId);
        });
    }

    async function loadConversations(selectFirst = true) {
        status.textContent = "";
        const response = await fetch("/api/conversations");
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Could not load conversations.");
        list.replaceChildren();
        const conversations = payload.conversations;
        conversations.forEach((conversation) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "thread-button";
            button.dataset.threadId = conversation.thread_id;
            const title = document.createElement("strong");
            title.textContent = conversation.thread_id.slice(0, 13);
            const timestamp = document.createElement("small");
            timestamp.textContent = new Date(conversation.updated_at).toLocaleString();
            button.append(title, timestamp);
            button.addEventListener("click", () => loadConversation(conversation.thread_id).catch(showError));
            list.append(button);
        });
        if (!conversations.length) {
            status.textContent = "No conversations yet. Start a new thread.";
            return;
        }
        if (selectFirst) await loadConversation(conversations[0].thread_id);
    }

    function showError(cause) {
        error.textContent = cause.message || "The chat request failed.";
    }

    document.getElementById("new-conversation").addEventListener("click", async () => {
        error.textContent = "";
        try {
            const response = await fetch("/api/conversations", { method: "POST" });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Could not create conversation.");
            await loadConversations(false);
            await loadConversation(payload.conversation.thread_id);
            input.focus();
        } catch (cause) {
            showError(cause);
        }
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        error.textContent = "";
        if (!activeThreadId || !form.reportValidity()) return;
        const sendButton = document.getElementById("send-message");
        const userText = input.value.trim();
        if (!userText) return;
        sendButton.disabled = true;
        input.disabled = true;
        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ thread_id: activeThreadId, message: userText }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Chat request failed.");
            input.value = "";
            renderHistory(payload.messages || []);
            mode.textContent = payload.mode === "groq" ? "GROQ CONNECTED" : "LOCAL FALLBACK";
            await loadConversations(false);
            await loadConversation(activeThreadId);
        } catch (cause) {
            showError(cause);
        } finally {
            input.disabled = false;
            sendButton.disabled = false;
            input.focus();
        }
    });

    loadConversations().catch(showError);
})();
