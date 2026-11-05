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
} from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // async function sendMessage() {
  //   const content = input.trim();

  //   if (!content || loading) return;

  //   const userMessage: Message = {
  //     role: "user",
  //     content,
  //   };

  //   const conversation = [...messages, userMessage];

  //   setMessages(conversation);
  //   setInput("");
  //   setError("");
  //   setLoading(true);

  //   try {
  //     const response = await fetch(`${API_URL}/api/chat`, {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json",
  //         Accept: "application/json",
  //       },
  //       body: JSON.stringify({
  //         messages: conversation,
  //       }),
  //     });

  //     const data = await response.json();

  //     if (!response.ok) {
  //       throw new Error(data.detail || "Something went wrong.");
  //     }

  //     const assistantMessage: Message = {
  //       role: "assistant",
  //       content: data.reply,
  //     };

  //     setMessages([...conversation, assistantMessage]);
  //   } catch (err) {
  //     setError(
  //       err instanceof Error
  //         ? err.message
  //         : "Unable to connect to RoboResearch AI."
  //     );
  //   } finally {
  //     setLoading(false);
  //     textareaRef.current?.focus();
  //   }
  // }



  
async function sendMessage() {
    const content = input.trim();

    if (!content || loading) return;

    const userMessage: Message = {
        role: "user",
        content,
    };

    const conversation = [...messages, userMessage];

    setMessages(conversation);
    setInput("");
    setError("");
    setLoading(true);

    const startedAt = performance.now();

    console.log("[RoboResearch AI] Request started");

    try {
        const response = await fetch(`${API_URL}/api/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({
                messages: conversation,
            }),
        });

        console.log(
            "[RoboResearch AI] HTTP response received:",
            response.status,
            "Elapsed:",
            ((performance.now() - startedAt) / 1000).toFixed(2),
            "seconds"
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Something went wrong.");
        }

        const assistantMessage: Message = {
            role: "assistant",
            content: data.reply,
        };

        setMessages([...conversation, assistantMessage]);

        console.log(
            "[RoboResearch AI] Generation completed in",
            ((performance.now() - startedAt) / 1000).toFixed(2),
            "seconds"
        );

    } catch (err) {
        console.error("[RoboResearch AI] Request failed:", err);

        setError(
            err instanceof Error
                ? `${err.name}: ${err.message}`
                : "Unable to connect to RoboResearch AI."
        );

    } finally {
        setLoading(false);
        textareaRef.current?.focus();
    }
}
  

  function newChat() {
    setMessages([]);
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

      {/* Sidebar */}
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

        <div className="mt-8">
          <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Conversations
          </p>

          {messages.length > 0 ? (
            <div className="flex items-center gap-2 rounded-lg bg-zinc-800 px-3 py-3 text-sm">
              <MessageSquare size={16} />
              <span className="truncate">Current conversation</span>

              <button
                onClick={newChat}
                className="ml-auto text-zinc-500 hover:text-red-400"
                title="Clear conversation"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ) : (
            <p className="px-2 text-sm text-zinc-600">
              No conversations yet
            </p>
          )}
        </div>

        <div className="mt-auto rounded-xl border border-zinc-800 bg-zinc-950 p-4">
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

      {/* Main area */}
      <main className="flex min-w-0 flex-1 flex-col">

        {/* Header */}
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

        {/* Messages */}
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
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex, rehypeHighlight]}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
                                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex gap-4">
                    <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-600">
                      <Bot size={19} />
                    </div>

                    <div className="pt-2 text-sm text-zinc-400">
                      Thinking...
                    </div>
                  </div>
                )}

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

        {/* Input */}
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
