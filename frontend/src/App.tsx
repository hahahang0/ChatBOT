
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import {
  Bot,
  Plus,
  Send,
  User,
  Menu,
  Trash2,
  Sparkles,
  MessageSquare,
  Pencil,
} from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null
  );

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // Fetch all saved conversations from SQLite.
  // async function fetchConversations() {
  //   try {
  //     const response = await fetch(`${API_URL}/api/conversations`);

  //     if (!response.ok) {
  //       throw new Error("Unable to load saved conversations.");
  //     }

  //     const data: Conversation[] = await response.json();
  //     setConversations(data);
  //   } catch (err) {
  //     console.error("[RoboResearch AI] Failed to load conversations:", err);
  //   }
  // }


  
async function fetchConversations() {
  try {
    const response = await fetch(`${API_URL}/api/conversations`);

    if (!response.ok) {
      throw new Error("Unable to load saved conversations.");
    }

    const data = await response.json();

    setConversations(data.conversations ?? []);
  } catch (err) {
    console.error("[RoboResearch AI] Failed to load conversations:", err);
  }
}


  // Load conversation history when the application opens.
  useEffect(() => {
    fetchConversations();
  }, []);

  // Cancel the current request before switching conversations.
  function cancelActiveRequest() {
    requestIdRef.current++;

    controllerRef.current?.abort();
    controllerRef.current = null;

    setLoading(false);
  }

  // Generate a readable title from the first user message.
  function generateConversationTitle(content: string) {
    const cleaned = content.replace(/\s+/g, " ").trim();

    if (cleaned.length <= 55) {
      return cleaned;
    }

    return cleaned.slice(0, 55).trimEnd() + "...";
  }

  // Create a new conversation in the backend.
  async function createConversation(title: string, signal: AbortSignal) {
    const response = await fetch(`${API_URL}/api/conversations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ title }),
      signal,
    });

    if (!response.ok) {
      let message = "Unable to create a new conversation.";

      try {
        const data = await response.json();
        message = data.detail || message;
      } catch {
        // Keep default error.
      }

      throw new Error(message);
    }

    return response.json();
  }

  // Open an existing conversation.
  async function openConversation(conversationId: string) {
    if (conversationId === activeConversationId && !loading) {
      return;
    }

    cancelActiveRequest();

    const requestId = requestIdRef.current;
    const controller = new AbortController();

    controllerRef.current = controller;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_URL}/api/conversations/${conversationId}`,
        {
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        throw new Error("Unable to open this conversation.");
      }

      const data = await response.json();

      if (requestId !== requestIdRef.current) return;

      const restoredMessages: Message[] = data.messages.map(
        (message: Message) => ({
          role: message.role,
          content: message.content,
        })
      );

      setMessages(restoredMessages);
      setActiveConversationId(conversationId);

      textareaRef.current?.focus();
    } catch (err) {
      if (controller.signal.aborted) return;

      setError(
        err instanceof Error
          ? err.message
          : "Unable to load this conversation."
      );
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
        controllerRef.current = null;
      }
    }
  }

  // Rename an existing conversation.
  async function renameConversation(conversation: Conversation) {
    const newTitle = window.prompt(
      "Enter a new conversation title:",
      conversation.title
    );

    if (!newTitle || !newTitle.trim()) return;

    try {
      const response = await fetch(
        `${API_URL}/api/conversations/${conversation.id}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: newTitle.trim(),
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Unable to rename this conversation.");
      }

      await fetchConversations();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to rename this conversation."
      );
    }
  }

  // Delete a conversation permanently.
  async function deleteConversation(conversationId: string) {
    const confirmed = window.confirm(
      "Are you sure you want to delete this conversation? This action cannot be undone."
    );

    if (!confirmed) return;

    try {
      const response = await fetch(
        `${API_URL}/api/conversations/${conversationId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error("Unable to delete this conversation.");
      }

      if (activeConversationId === conversationId) {
        cancelActiveRequest();
        setMessages([]);
        setActiveConversationId(null);
        setInput("");
      }

      await fetchConversations();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to delete this conversation."
      );
    }
  }

  async function sendMessage() {
    const content = input.trim();

    if (!content || loading) return;

    const userMessage: Message = {
      role: "user",
      content,
    };

    const conversation = [...messages, userMessage];
    const assistantIndex = conversation.length;

    const requestId = ++requestIdRef.current;
    const controller = new AbortController();

    controllerRef.current = controller;

    setInput("");
    setError("");
    setLoading(true);

    const startedAt = performance.now();

    console.log("[RoboResearch AI] Streaming request started");

    try {
      // Create a database conversation only when the first message is sent.
      let conversationId = activeConversationId;

      if (!conversationId) {
        const created = await createConversation(
          generateConversationTitle(content),
          controller.signal
        );

        if (requestId !== requestIdRef.current) return;

        conversationId = created.id;

        if (!conversationId) {
          throw new Error("The backend did not return a conversation ID.");
        }

        setActiveConversationId(conversationId);
        void fetchConversations();
      }

      if (requestId !== requestIdRef.current) return;

      // Add the user message and an empty assistant response.
      setMessages([
        ...conversation,
        {
          role: "assistant",
          content: "",
        },
      ]);

      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/x-ndjson",
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          messages: conversation,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        let errorMessage = "Something went wrong.";

        try {
          const data = await response.json();
          errorMessage = data.detail || errorMessage;
        } catch {
          // Keep default error.
        }

        throw new Error(errorMessage);
      }

      if (!response.body) {
        throw new Error("Streaming is not supported by this response.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let receivedDone = false;

      // Process one NDJSON event.
      function processEvent(line: string) {
        if (!line.trim()) return;

        const event = JSON.parse(line);

        if (event.type === "token") {
          if (requestId !== requestIdRef.current) return;

          setMessages((previous) =>
            previous.map((message, index) =>
              index === assistantIndex
                ? {
                    ...message,
                    content: message.content + event.content,
                  }
                : message
            )
          );
        }

        if (event.type === "done") {
          receivedDone = true;

          console.log(
            "[RoboResearch AI] Generation completed in",
            ((performance.now() - startedAt) / 1000).toFixed(2),
            "seconds"
          );

          console.log("[RoboResearch AI] Model:", event.model);
          console.log("[RoboResearch AI] Finish reason:", event.finish_reason);
          console.log("[RoboResearch AI] Generated tokens:", event.eval_count);
          console.log("[RoboResearch AI] Prompt tokens:", event.prompt_eval_count);
        }

        if (event.type === "error") {
          throw new Error(event.message || "Streaming failed.");
        }
      }

      try {
        while (true) {
          const { value, done } = await reader.read();

          if (done) break;

          if (requestId !== requestIdRef.current) break;

          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            processEvent(line);
          }
        }

        // Process any remaining data after the stream closes.
        buffer += decoder.decode();

        if (buffer.trim() && requestId === requestIdRef.current) {
          processEvent(buffer);
        }

        if (!receivedDone && requestId === requestIdRef.current) {
          throw new Error(
            "The response ended before generation completed."
          );
        }
      } finally {
        reader.releaseLock();
      }

      if (receivedDone && requestId === requestIdRef.current) {
        void fetchConversations();
      }
    } catch (err) {
      if (controller.signal.aborted) return;

      console.error("[RoboResearch AI] Streaming failed:", err);

      setError(
        err instanceof Error
          ? `${err.name}: ${err.message}`
          : "Unable to connect to RoboResearch AI."
      );
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
        controllerRef.current = null;
        textareaRef.current?.focus();
      }
    }
  }

  function newChat() {
    cancelActiveRequest();

    setMessages([]);
    setActiveConversationId(null);
    setInput("");
    setError("");

    textareaRef.current?.focus();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      <aside
        className={`${
          sidebarOpen ? "flex" : "hidden"
        } w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/80 p-4`}
      >
        <div className="mb-8 flex items-center gap-3 px-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600">
            <Bot size={23} />
          </div>

          <div>
            <h1 className="font-semibold">RoboResearch AI</h1>
            <p className="text-xs text-zinc-500">
              Your robotics assistant
            </p>
          </div>
        </div>

        <button
          onClick={newChat}
          className="flex items-center gap-3 rounded-xl border border-zinc-700 px-4 py-3 text-sm transition hover:bg-zinc-800"
        >
          <Plus size={18} />
          New chat
        </button>

        <div className="mt-8 min-h-0 flex-1 overflow-y-auto">
          <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Conversations
          </p>

          {conversations.length > 0 ? (
            <div className="space-y-2">
              {conversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`group flex items-center gap-2 rounded-lg px-3 py-3 text-sm transition ${
                    activeConversationId === conversation.id
                      ? "bg-zinc-800 text-white"
                      : "text-zinc-400 hover:bg-zinc-800/70 hover:text-white"
                  }`}
                >
                  <button
                    onClick={() => openConversation(conversation.id)}
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    title={conversation.title}
                  >
                    <MessageSquare size={16} className="shrink-0" />

                    <span className="truncate">
                      {conversation.title}
                    </span>
                  </button>

                  <button
                    onClick={() => renameConversation(conversation)}
                    className="shrink-0 text-zinc-500 opacity-0 transition hover:text-blue-400 group-hover:opacity-100"
                    title="Rename conversation"
                  >
                    <Pencil size={14} />
                  </button>

                  <button
                    onClick={() => deleteConversation(conversation.id)}
                    className="shrink-0 text-zinc-500 opacity-0 transition hover:text-red-400 group-hover:opacity-100"
                    title="Delete conversation"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="px-2 text-sm text-zinc-600">
              No conversations yet
            </p>
          )}
        </div>

        <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm">
            <Sparkles size={16} className="text-blue-400" />
            Local AI
          </div>

          <p className="text-xs leading-relaxed text-zinc-500">
            Powered by Qwen running locally through Ollama.
            Your prompts are processed on your computer.
          </p>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-zinc-800 px-5">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-white"
              title="Toggle sidebar"
            >
              <Menu size={20} />
            </button>

            <div>
              <h2 className="text-sm font-semibold">
                RoboResearch AI
              </h2>
              <p className="text-xs text-zinc-500">
                Qwen3.5 2B · Local model
              </p>
            </div>
          </div>

          <div className="rounded-full border border-emerald-800/60 bg-emerald-950/40 px-3 py-1 text-xs text-emerald-400">
            Local engine
          </div>
        </header>

        <section className="flex-1 overflow-y-auto px-4 py-8">
          <div className="mx-auto max-w-3xl">
            {messages.length === 0 ? (
              <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600/15 text-blue-400">
                  <Bot size={36} />
                </div>

                <h2 className="mb-3 text-3xl font-semibold tracking-tight">
                  What are we exploring today?
                </h2>

                <p className="mb-10 max-w-lg text-sm leading-relaxed text-zinc-500">
                  Ask me about robotics, control systems,
                  dynamics, motion planning, machine learning,
                  UAVs, spacecraft, or anything else.
                </p>

                <div className="grid w-full gap-3 sm:grid-cols-2">
                  {[
                    "Explain PID control with a quadrotor example.",
                    "What is the difference between A* and RRT*?",
                    "Explain forward and inverse kinematics.",
                    "How does a spacecraft maintain its attitude?",
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => setInput(suggestion)}
                      className="rounded-xl border border-zinc-800 p-4 text-left text-sm text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-900"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-8">
                {messages.map((message, index) => (
                  <div key={index} className="flex gap-4">
                    <div
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${
                        message.role === "assistant"
                          ? "bg-blue-600"
                          : "bg-zinc-700"
                      }`}
                    >
                      {message.role === "assistant" ? (
                        <Bot size={19} />
                      ) : (
                        <User size={19} />
                      )}
                    </div>

                    <div className="min-w-0 flex-1 pt-1">
                      <p className="mb-2 text-sm font-semibold">
                        {message.role === "assistant"
                          ? "RoboResearch AI"
                          : "You"}
                      </p>

                      <div className="markdown-content text-sm leading-7 text-zinc-300">
                        {message.content ? (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex, rehypeHighlight]}
                          >
                            {message.content}
                          </ReactMarkdown>
                        ) : loading && index === messages.length - 1 ? (
                          <span className="text-zinc-400">
                            Thinking...
                          </span>
                        ) : null}
                      </div>
                    </div>
                  </div>
                ))}

                <div ref={bottomRef} />
              </div>
            )}

            {error && (
              <div className="mt-6 rounded-xl border border-red-900 bg-red-950/30 p-4 text-sm text-red-300">
                <strong>Request failed:</strong> {error}
              </div>
            )}
          </div>
        </section>

        <footer className="shrink-0 px-4 pb-6 pt-3">
          <div className="mx-auto max-w-3xl">
            <div className="rounded-2xl border border-zinc-700 bg-zinc-900 p-3 shadow-lg focus-within:border-zinc-500">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message RoboResearch AI..."
                rows={2}
                disabled={loading}
                className="max-h-40 w-full resize-none bg-transparent px-2 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
              />

              <div className="flex items-center justify-between px-2 pt-2">
                <p className="text-xs text-zinc-600">
                  Enter to send · Shift + Enter for a new line
                </p>

                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || loading}
                  className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
                  title="Send message"
                >
                  <Send size={18} />
                </button>
              </div>
            </div>

            <p className="mt-3 text-center text-xs text-zinc-600">
              RoboResearch AI can make mistakes. Verify important information.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
