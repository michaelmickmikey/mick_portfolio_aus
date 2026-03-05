(function () {
  const fab = document.getElementById("chatFab");
  const panel = document.getElementById("chatPanel");
  const closeBtn = document.getElementById("chatClose");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");
  const messages = document.getElementById("chatMessages");

  let thread = []; // simple in-memory chat history for this page load

  function openChat() {
    panel.classList.add("open");
    panel.setAttribute("aria-hidden", "false");
    input.focus();
    if (messages.childElementCount === 0) {
      addBot("Hey — I’m Mick’s portfolio assistant. Want a quick rundown of the dissertation, projects, or experience?");
    }
  }

  function closeChat() {
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
  }

  function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function addMsg(role, text) {
    const row = document.createElement("div");
    row.className = `chat-row ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  function addUser(text) { addMsg("user", text); }
  function addBot(text) { addMsg("bot", text); }

  async function sendToServer(userText) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userText,
        history: thread.slice(-8) // keep it short
      })
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || `HTTP ${res.status}`);
    }

    return res.json();
  }

  fab?.addEventListener("click", openChat);
  closeBtn?.addEventListener("click", closeChat);

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const userText = (input.value || "").trim();
    if (!userText) return;

    input.value = "";
    addUser(userText);

    // store in local history
    thread.push({ role: "user", content: userText });

    // optimistic typing indicator
    const typing = document.createElement("div");
    typing.className = "chat-row bot";
    typing.innerHTML = `<div class="chat-bubble chat-typing">Typing…</div>`;
    messages.appendChild(typing);
    scrollToBottom();

    try {
      const data = await sendToServer(userText);
      typing.remove();
      addBot(data.reply || "No reply received.");
      thread.push({ role: "assistant", content: data.reply || "" });
    } catch (err) {
      typing.remove();
      addBot("Sorry — something went wrong calling the AI. Try again in a moment.");
      console.error(err);
    }
  });
})();