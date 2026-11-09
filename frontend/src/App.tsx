
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
  FileText,
  Upload,
  LoaderCircle,
  CheckCircle2,
  AlertCircle,
  Library,
  BookOpen,
  Cpu,
  RefreshCw,
  Paperclip,
  X,
} from "lucide-react";

// --------------------------------------------------
// TYPES
// --------------------------------------------------

type Document = {
  id: string;
  filename: string;
  file_size: number;
  page_count: number;
  status: "processing" | "ready" | "failed";
  error_message?: string | null;
  created_at: string;
};

type Source = {
  document_id: string;
  filename: string;
  page_number: number;
  similarity: number;
  excerpt: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  truncated?: boolean;
};

type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type ModelsResponse = {
  models: string[];
  default_model: string;
};

type StreamEvent = {
  type: "token" | "done" | "error" | string;
  content?: string;
  message?: string;
  model?: string;
  finish_reason?: string;
  eval_count?: number;
  prompt_eval_count?: number;
  sources?: Source[];
  truncated?: boolean;
};

// --------------------------------------------------
// CONFIGURATION
// --------------------------------------------------

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const DEFAULT_MODEL = "openai/gpt-oss-120b";

function isEmbeddingModel(model: string) {
  return /embed|all-minilm|nomic-embed|mxbai|bge-|snowflake-arctic/i.test(
    model
  );
}

// --------------------------------------------------
// STREAMING JSON PARSER
// --------------------------------------------------

function parseStreamEvents(rawLine: string): StreamEvent[] {
  let line = rawLine.trim();

  if (!line) {
    return [];
  }

  if (line.startsWith("data:")) {
    line = line.slice(5).trim();
  }

  if (!line || line === "[DONE]") {
    return [];
  }

  const events: StreamEvent[] = [];

  let depth = 0;
  let start = -1;
  let insideString = false;
  let escaped = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (insideString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        insideString = false;
      }

      continue;
    }

    if (char === '"') {
      insideString = true;
      continue;
    }

    if (char === "{") {
      if (depth === 0) {
        start = i;
      }

      depth++;
      continue;
    }

    if (char === "}") {
      depth--;

      if (depth < 0) {
        throw new Error(
          "Invalid streaming JSON: unexpected closing brace."
        );
      }

      if (depth === 0 && start !== -1) {
        const jsonText = line.slice(start, i + 1);

        try {
          events.push(JSON.parse(jsonText) as StreamEvent);
        } catch {
          throw new Error(
            "Unable to parse a streaming event from the server."
          );
        }

        start = -1;
      }
    }
  }

  if (depth !== 0 || insideString) {
    throw new Error(
      "Incomplete JSON event received from the server. Check the backend streaming response."
    );
  }

  if (events.length === 0 && line.length > 0) {
    throw new Error(
      `Unexpected streaming response: ${line.slice(0, 120)}`
    );
  }

  return events;
}

// --------------------------------------------------
// MAIN APP
// --------------------------------------------------

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelError, setModelError] = useState("");

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] =
    useState<string | null>(null);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [documentError, setDocumentError] = useState("");
  const [attachedDocument, setAttachedDocument] =
    useState<Document | null>(null);

  const chatFileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  // ------------------------------------------------
  // SCROLL TO LATEST MESSAGE
  // ------------------------------------------------

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // ------------------------------------------------
  // FETCH AVAILABLE MODELS
  // ------------------------------------------------

  async function fetchModels() {
    setModelsLoading(true);
    setModelError("");

    try {
      const response = await fetch(`${API_URL}/api/models`);

      if (!response.ok) {
        let message = "Unable to retrieve available models.";

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          // Keep default error.
        }

        throw new Error(message);
      }

      const data: ModelsResponse = await response.json();

      const availableModels = (data.models ?? []).filter(
        (model) => !isEmbeddingModel(model)
      );

      setModels(availableModels);

      setSelectedModel((currentModel) => {
        if (availableModels.includes(currentModel)) {
          return currentModel;
        }

        if (
          data.default_model &&
          availableModels.includes(data.default_model)
        ) {
          return data.default_model;
        }

        return availableModels[0] ?? "";
      });

      setModelError(
        availableModels.length === 0
          ? "No conversational models were returned by the backend."
          : ""
      );
    } catch (err) {
      setModelError(
        err instanceof Error
          ? err.message
          : "Unable to load available models."
      );
    } finally {
      setModelsLoading(false);
    }
  }

  // ------------------------------------------------
  // FETCH CONVERSATIONS
  // ------------------------------------------------

  async function fetchConversations() {
    try {
      const response = await fetch(`${API_URL}/api/conversations`);

      if (!response.ok) {
        throw new Error("Unable to load saved conversations.");
      }

      const data = await response.json();

      setConversations(data.conversations ?? []);
    } catch (err) {
      console.error(
        "[RoboResearch AI] Failed to load conversations:",
        err
      );
    }
  }

  // ------------------------------------------------
  // CREATE CONVERSATION
  // ------------------------------------------------

  async function createConversation(
    title: string,
    signal: AbortSignal
  ) {
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

  // ------------------------------------------------
  // OPEN CONVERSATION
  // ------------------------------------------------

  async function openConversation(conversationId: string) {
    if (
      conversationId === activeConversationId &&
      !loading
    ) {
      return;
    }

    cancelActiveRequest();

    const requestId = requestIdRef.current;
    const controller = new AbortController();

    controllerRef.current = controller;

    setLoading(true);
    setError("");
    setAttachedDocument(null);

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

      if (requestId !== requestIdRef.current) {
        return;
      }

      const restoredMessages: Message[] = (
        data.messages ?? []
      ).map((message: Message) => ({
        role: message.role,
        content: message.content,
        sources: message.sources ?? [],
      }));

      setMessages(restoredMessages);
      setActiveConversationId(conversationId);

      textareaRef.current?.focus();
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }

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

  // ------------------------------------------------
  // RENAME CONVERSATION
  // ------------------------------------------------

  async function renameConversation(conversation: Conversation) {
    const newTitle = window.prompt(
      "Enter a new conversation title:",
      conversation.title
    );

    if (!newTitle || !newTitle.trim()) {
      return;
    }

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

  // ------------------------------------------------
  // DELETE CONVERSATION
  // ------------------------------------------------

  async function deleteConversation(conversationId: string) {
    const confirmed = window.confirm(
      "Are you sure you want to delete this conversation? This action cannot be undone."
    );

    if (!confirmed) {
      return;
    }

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
        setAttachedDocument(null);
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

  // ------------------------------------------------
  // FETCH DOCUMENTS
  // ------------------------------------------------

  async function fetchDocuments() {
    try {
      const response = await fetch(`${API_URL}/api/documents`);

      if (!response.ok) {
        throw new Error("Unable to load documents.");
      }

      const data = await response.json();

      setDocuments(data.documents ?? []);
    } catch (err) {
      console.error(
        "[RoboResearch AI] Failed to load documents:",
        err
      );
    }
  }

  // ------------------------------------------------
  // SELECT A LIBRARY DOCUMENT FOR CHAT
  // ------------------------------------------------

  function selectDocumentForChat(document: Document) {
    if (document.status !== "ready") {
      setDocumentError(
        "This PDF is not ready yet. Please wait until processing is complete."
      );
      return;
    }

    setAttachedDocument(document);
    setDocumentError("");

    textareaRef.current?.focus();

    console.log(
      "[RoboResearch AI] Document selected for chat:",
      document.id,
      document.filename
    );
  }

  // ------------------------------------------------
  // UPLOAD PDF AND RETURN DOCUMENT DETAILS
  // ------------------------------------------------

  async function uploadPdf(file: File): Promise<Document> {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      throw new Error("Please select a PDF file.");
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(
      `${API_URL}/api/documents/upload`,
      {
        method: "POST",
        body: formData,
      }
    );

    let data: {
      document?: Document;
      detail?: string;
    };

    try {
      data = await response.json();
    } catch {
      throw new Error(
        "The server returned an invalid upload response."
      );
    }

    if (!response.ok) {
      throw new Error(
        data.detail || "Unable to upload this document."
      );
    }

    if (!data.document?.id) {
      throw new Error(
        "The server uploaded the PDF but did not return a valid document ID."
      );
    }

    return data.document;
  }

  // ------------------------------------------------
  // ATTACH PDF DIRECTLY TO CHAT
  // ------------------------------------------------

  async function attachDocumentToChat(file: File) {
    setUploading(true);
    setDocumentError("");

    try {
      const uploaded = await uploadPdf(file);

      if (uploaded.status !== "ready") {
        throw new Error(
          uploaded.error_message ||
            "The PDF has not finished processing."
        );
      }

      setAttachedDocument(uploaded);

      await fetchDocuments();

      console.log(
        "[RoboResearch AI] PDF attached to chat:",
        uploaded.id,
        uploaded.filename
      );
    } catch (err) {
      setDocumentError(
        err instanceof Error
          ? err.message
          : "Document upload failed."
      );
    } finally {
      setUploading(false);

      if (chatFileInputRef.current) {
        chatFileInputRef.current.value = "";
      }
    }
  }

  // ------------------------------------------------
  // UPLOAD PDF TO RESEARCH LIBRARY
  // ------------------------------------------------

  async function uploadDocument(file: File) {
    setUploading(true);
    setDocumentError("");

    try {
      const uploaded = await uploadPdf(file);

      // IMPORTANT:
      // A PDF uploaded through the Research Library
      // is now also attached to the current chat.
      if (uploaded.status === "ready") {
        setAttachedDocument(uploaded);

        console.log(
          "[RoboResearch AI] Library PDF automatically attached:",
          uploaded.id,
          uploaded.filename
        );
      } else {
        setDocumentError(
          uploaded.error_message ||
            "The PDF was uploaded but is not ready for chat yet."
        );
      }

      await fetchDocuments();
    } catch (err) {
      setDocumentError(
        err instanceof Error
          ? err.message
          : "Document upload failed."
      );
    } finally {
      setUploading(false);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  // ------------------------------------------------
  // DELETE DOCUMENT
  // ------------------------------------------------

  async function deleteDocument(documentId: string) {
    const confirmed = window.confirm(
      "Are you sure you want to delete this document? Its extracted text and embeddings will also be removed."
    );

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(
        `${API_URL}/api/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        let message = "Unable to delete this document.";

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          // Keep default error.
        }

        throw new Error(message);
      }

      if (attachedDocument?.id === documentId) {
        setAttachedDocument(null);
      }

      await fetchDocuments();
    } catch (err) {
      setDocumentError(
        err instanceof Error
          ? err.message
          : "Unable to delete this document."
      );
    }
  }

  // ------------------------------------------------
  // FORMAT FILE SIZE
  // ------------------------------------------------

  function formatFileSize(bytes: number) {
    if (bytes < 1024) {
      return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // ------------------------------------------------
  // INITIAL DATA LOADING
  // ------------------------------------------------

  useEffect(() => {
    void fetchConversations();
    void fetchDocuments();
    void fetchModels();
  }, []);

  // ------------------------------------------------
  // CANCEL ACTIVE REQUEST
  // ------------------------------------------------

  function cancelActiveRequest() {
    requestIdRef.current++;

    controllerRef.current?.abort();
    controllerRef.current = null;

    setLoading(false);
  }

  // ------------------------------------------------
  // GENERATE CONVERSATION TITLE
  // ------------------------------------------------

  function generateConversationTitle(content: string) {
    const cleaned = content.replace(/\s+/g, " ").trim();

    if (cleaned.length <= 55) {
      return cleaned;
    }

    return cleaned.slice(0, 55).trimEnd() + "...";
  }

  // ------------------------------------------------
  // HANDLE STREAM EVENTS
  // ------------------------------------------------

  function handleStreamEvent(
    event: StreamEvent,
    requestId: number,
    assistantIndex: number,
    startedAt: number
  ) {
    if (requestId !== requestIdRef.current) {
      return false;
    }

    if (event.type === "token") {
      setMessages((previous) =>
        previous.map((message, index) =>
          index === assistantIndex
            ? {
                ...message,
                content: message.content + (event.content ?? ""),
              }
            : message
        )
      );

      return false;
    }

    if (event.type === "done") {
      console.log(
        "[RoboResearch AI] Generation completed in",
        ((performance.now() - startedAt) / 1000).toFixed(2),
        "seconds"
      );

      console.log("[RoboResearch AI] Model:", event.model);
      console.log(
        "[RoboResearch AI] Finish reason:",
        event.finish_reason
      );
      console.log(
        "[RoboResearch AI] Generated tokens:",
        event.eval_count
      );
      console.log(
        "[RoboResearch AI] Prompt tokens:",
        event.prompt_eval_count
      );

      const receivedSources: Source[] = Array.isArray(
        event.sources
      )
        ? event.sources
        : [];

      const wasTruncated =
        event.truncated === true ||
        event.finish_reason === "length";

      setMessages((previous) =>
        previous.map((message, index) =>
          index === assistantIndex
            ? {
                ...message,
                sources: receivedSources,
                truncated: wasTruncated,
              }
            : message
        )
      );

      if (wasTruncated) {
        console.warn(
          "[RoboResearch AI] Response reached the output token limit."
        );
      }

      console.log(
        "[RoboResearch AI] Retrieved sources:",
        receivedSources.length
      );

      return true;
    }

    if (event.type === "error") {
      throw new Error(event.message || "Streaming failed.");
    }

    console.warn(
      "[RoboResearch AI] Unknown stream event:",
      event
    );

    return false;
  }

  // ------------------------------------------------
  // SEND MESSAGE
  // ------------------------------------------------

  async function sendMessage() {
    const content =
      input.trim() ||
      (attachedDocument
        ? `Please analyze the attached research paper: ${attachedDocument.filename}`
        : "");

    if (!content || loading) {
      return;
    }

    if (!selectedModel) {
      setError("Please select an available AI model.");
      return;
    }

    if (
      attachedDocument &&
      attachedDocument.status !== "ready"
    ) {
      setError(
        "The attached PDF is not ready yet. Please wait for processing to finish."
      );
      return;
    }

    const selectedDocument = attachedDocument;

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
    console.log("[RoboResearch AI] Selected model:", selectedModel);
    console.log(
      "[RoboResearch AI] Attached document ID:",
      selectedDocument?.id ?? "None"
    );
    console.log(
      "[RoboResearch AI] Attached document:",
      selectedDocument?.filename ?? "None"
    );

    try {
      let conversationId = activeConversationId;

      if (!conversationId) {
        const created = await createConversation(
          generateConversationTitle(content),
          controller.signal
        );

        if (requestId !== requestIdRef.current) {
          return;
        }

        conversationId = created.id;

        if (!conversationId) {
          throw new Error(
            "The backend did not return a conversation ID."
          );
        }

        setActiveConversationId(conversationId);

        void fetchConversations();
      }

      if (requestId !== requestIdRef.current) {
        return;
      }

      setMessages([
        ...conversation,
        {
          role: "assistant",
          content: "",
          sources: [],
          truncated: false,
        },
      ]);

      const requestBody = {
        conversation_id: conversationId,
        messages: conversation,
        model: selectedModel,

        // The selected PDF ID is explicitly sent
        // with every chat request.
        document_id: selectedDocument?.id ?? null,
      };

      console.log(
        "[RoboResearch AI] Sending document_id:",
        requestBody.document_id
      );

      const response = await fetch(
        `${API_URL}/api/chat/stream`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/x-ndjson",
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        }
      );

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
        throw new Error(
          "Streaming is not supported by this response."
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let receivedDone = false;

      function processLine(line: string) {
        const events = parseStreamEvents(line);

        for (const event of events) {
          const completed = handleStreamEvent(
            event,
            requestId,
            assistantIndex,
            startedAt
          );

          if (completed) {
            receivedDone = true;
          }
        }
      }

      try {
        while (true) {
          const { value, done } = await reader.read();

          if (done) {
            break;
          }

          if (requestId !== requestIdRef.current) {
            break;
          }

          buffer += decoder.decode(value, {
            stream: true,
          });

          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (requestId !== requestIdRef.current) {
              break;
            }

            processLine(line);
          }
        }

        buffer += decoder.decode();

        if (
          buffer.trim() &&
          requestId === requestIdRef.current
        ) {
          processLine(buffer);
        }

        if (
          !receivedDone &&
          requestId === requestIdRef.current
        ) {
          throw new Error(
            "The response ended before generation completed. Check the backend streaming response."
          );
        }
      } finally {
        reader.releaseLock();
      }

      if (
        receivedDone &&
        requestId === requestIdRef.current
      ) {
        // Detach after a successful response.
        setAttachedDocument(null);

        void fetchConversations();
      }
    } catch (err) {
      if (controller.signal.aborted) {
        return;
      }

      console.error(
        "[RoboResearch AI] Streaming failed:",
        err
      );

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

  // ------------------------------------------------
  // NEW CHAT
  // ------------------------------------------------

  function newChat() {
    cancelActiveRequest();

    setMessages([]);
    setActiveConversationId(null);
    setInput("");
    setError("");
    setAttachedDocument(null);
    setDocumentError("");

    textareaRef.current?.focus();
  }

  // ------------------------------------------------
  // KEYBOARD HANDLING
  // ------------------------------------------------

  function handleKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      void sendMessage();
    }
  }

  // ------------------------------------------------
  // RENDER
  // ------------------------------------------------

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      {/* SIDEBAR */}
      <aside
        className={`${
          sidebarOpen ? "flex" : "hidden"
        } w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-900/80 p-4`}
      >
        {/* BRAND */}
        <div className="mb-8 flex items-center gap-3 px-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600">
            <Bot size={23} />
          </div>

          <div>
            <h1 className="font-semibold">
              RoboResearch AI
            </h1>

            <p className="text-xs text-zinc-500">
              Your robotics assistant
            </p>
          </div>
        </div>

        {/* NEW CHAT */}
        <button
          onClick={newChat}
          className="flex items-center gap-3 rounded-xl border border-zinc-700 px-4 py-3 text-sm transition hover:bg-zinc-800"
        >
          <Plus size={18} />
          New chat
        </button>

        {/* CONVERSATIONS */}
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
                    onClick={() =>
                      void openConversation(conversation.id)
                    }
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    title={conversation.title}
                  >
                    <MessageSquare
                      size={16}
                      className="shrink-0"
                    />

                    <span className="truncate">
                      {conversation.title}
                    </span>
                  </button>

                  <button
                    onClick={() =>
                      void renameConversation(conversation)
                    }
                    className="shrink-0 text-zinc-500 opacity-0 transition hover:text-blue-400 group-hover:opacity-100"
                    title="Rename conversation"
                  >
                    <Pencil size={14} />
                  </button>

                  <button
                    onClick={() =>
                      void deleteConversation(conversation.id)
                    }
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

        {/* RESEARCH LIBRARY */}
        <div className="mt-6 min-h-0 flex-1 overflow-y-auto">
          <div className="mb-3 flex items-center justify-between px-2">
            <div className="flex items-center gap-2">
              <Library
                size={15}
                className="text-blue-400"
              />

              <p className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Research Library
              </p>
            </div>

            <span className="text-xs text-zinc-500">
              {documents.length}
            </span>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];

              if (file) {
                void uploadDocument(file);
              }
            }}
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-zinc-700 px-3 py-3 text-sm text-zinc-300 transition hover:border-blue-500 hover:bg-blue-500/5 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {uploading ? (
              <LoaderCircle
                size={17}
                className="animate-spin"
              />
            ) : (
              <Upload size={17} />
            )}

            {uploading
              ? "Processing PDF..."
              : "Upload research paper"}
          </button>

          {documentError && (
            <div className="mb-3 rounded-lg border border-red-900 bg-red-950/30 p-2 text-xs text-red-300">
              {documentError}
            </div>
          )}

          {documents.length > 0 ? (
            <div className="space-y-2">
              {documents.map((document) => {
                const isAttached =
                  attachedDocument?.id === document.id;

                return (
                  <div
                    key={document.id}
                    className={`group rounded-lg border p-3 ${
                      isAttached
                        ? "border-blue-700/70 bg-blue-950/20"
                        : "border-zinc-800 bg-zinc-950/60"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <FileText
                        size={17}
                        className="mt-0.5 shrink-0 text-blue-400"
                      />

                      <div className="min-w-0 flex-1">
                        <p
                          className="truncate text-xs font-medium text-zinc-300"
                          title={document.filename}
                        >
                          {document.filename}
                        </p>

                        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                          {document.status === "ready" && (
                            <span className="flex items-center gap-1 text-emerald-400">
                              <CheckCircle2 size={12} />
                              Ready
                            </span>
                          )}

                          {document.status === "processing" && (
                            <span className="flex items-center gap-1 text-amber-400">
                              <LoaderCircle
                                size={12}
                                className="animate-spin"
                              />
                              Processing
                            </span>
                          )}

                          {document.status === "failed" && (
                            <span
                              className="flex items-center gap-1 text-red-400"
                              title={
                                document.error_message ||
                                "Processing failed"
                              }
                            >
                              <AlertCircle size={12} />
                              Failed
                            </span>
                          )}

                          <span className="text-zinc-600">
                            {formatFileSize(document.file_size)}
                          </span>

                          <span className="text-zinc-600">
                            {document.page_count} pages
                          </span>
                        </div>

                        {/* SELECT DOCUMENT FOR CHAT */}
                        <button
                          type="button"
                          onClick={() =>
                            selectDocumentForChat(document)
                          }
                          disabled={
                            document.status !== "ready" ||
                            uploading ||
                            loading
                          }
                          className={`mt-3 rounded-md px-2.5 py-1.5 text-xs transition disabled:cursor-not-allowed disabled:opacity-40 ${
                            isAttached
                              ? "bg-blue-600/20 text-blue-300"
                              : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white"
                          }`}
                        >
                          {isAttached
                            ? "Selected for chat"
                            : "Use in chat"}
                        </button>
                      </div>

                      <button
                        onClick={() =>
                          void deleteDocument(document.id)
                        }
                        className="shrink-0 text-zinc-600 opacity-0 transition hover:text-red-400 group-hover:opacity-100"
                        title="Delete document"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="px-2 py-3 text-center text-xs text-zinc-600">
              No research papers uploaded yet.
            </p>
          )}
        </div>

        {/* CLOUD AI STATUS */}
        <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm">
            <Sparkles
              size={16}
              className="text-blue-400"
            />
            Cloud AI
          </div>

          <p className="text-xs leading-relaxed text-zinc-500">
            Powered by {selectedModel || "Groq"} through the cloud API.
          </p>
        </div>
      </aside>

      {/* MAIN AREA */}
      <main className="flex min-w-0 flex-1 flex-col">
        {/* HEADER */}
        <header className="flex min-h-16 shrink-0 items-center justify-between gap-3 border-b border-zinc-800 px-5 py-2">
          <div className="flex min-w-0 items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="shrink-0 rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-white"
              title="Toggle sidebar"
            >
              <Menu size={20} />
            </button>

            <div className="min-w-0">
              <h2 className="text-sm font-semibold">
                RoboResearch AI
              </h2>

              <p className="truncate text-xs text-zinc-500">
                {selectedModel || "No model selected"} · Cloud model
              </p>
            </div>
          </div>

          {/* MODEL SELECTOR */}
          <div className="flex shrink-0 items-center gap-2">
            <div className="flex items-center gap-2 rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2">
              <Cpu
                size={16}
                className="shrink-0 text-blue-400"
              />

              <select
                value={selectedModel}
                onChange={(event) =>
                  setSelectedModel(event.target.value)
                }
                disabled={
                  modelsLoading ||
                  models.length === 0 ||
                  loading
                }
                className="max-w-40 cursor-pointer bg-transparent text-xs text-zinc-200 outline-none disabled:cursor-not-allowed disabled:opacity-50 sm:max-w-52"
                title="Select AI model"
              >
                {models.length === 0 && (
                  <option value={selectedModel}>
                    {modelsLoading
                      ? "Loading models..."
                      : "No models available"}
                  </option>
                )}

                {models.map((model) => (
                  <option
                    key={model}
                    value={model}
                    className="bg-zinc-900 text-zinc-200"
                  >
                    {model}
                  </option>
                ))}
              </select>

              <button
                onClick={() => void fetchModels()}
                disabled={modelsLoading || loading}
                className="ml-1 text-zinc-500 transition hover:text-blue-400 disabled:opacity-40"
                title="Refresh available models"
              >
                <RefreshCw
                  size={14}
                  className={
                    modelsLoading ? "animate-spin" : ""
                  }
                />
              </button>
            </div>

            <div className="hidden rounded-full border border-emerald-800/60 bg-emerald-950/40 px-3 py-1 text-xs text-emerald-400 sm:block">
              Cloud engine
            </div>
          </div>
        </header>

        {/* MODEL ERROR */}
        {modelError && (
          <div className="flex items-center justify-between gap-3 border-b border-amber-900/50 bg-amber-950/20 px-5 py-2">
            <p className="text-xs text-amber-300">
              <AlertCircle
                size={13}
                className="mr-2 inline"
              />
              {modelError}
            </p>

            <button
              onClick={() => void fetchModels()}
              className="shrink-0 text-xs text-amber-300 underline hover:text-amber-200"
            >
              Retry
            </button>
          </div>
        )}

        {/* CHAT MESSAGES */}
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
                  <div
                    key={index}
                    className="flex gap-4"
                  >
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
                            remarkPlugins={[
                              remarkGfm,
                              remarkMath,
                            ]}
                            rehypePlugins={[
                              rehypeKatex,
                              rehypeHighlight,
                            ]}
                          >
                            {message.content}
                          </ReactMarkdown>
                        ) : loading &&
                          index === messages.length - 1 ? (
                          <span className="text-zinc-400">
                            Thinking...
                          </span>
                        ) : null}
                      </div>

                      {/* TRUNCATION WARNING */}
                      {message.role === "assistant" &&
                        message.truncated && (
                          <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-800/60 bg-amber-950/30 p-3 text-xs leading-relaxed text-amber-300">
                            <AlertCircle
                              size={16}
                              className="mt-0.5 shrink-0"
                            />

                            <div>
                              <p className="font-semibold">
                                Response may be incomplete
                              </p>

                              <p className="mt-1 text-amber-400/80">
                                The model reached its maximum output
                                token limit before finishing its answer.
                                Ask it to continue or request a shorter
                                response.
                              </p>
                            </div>
                          </div>
                        )}

                      {/* RETRIEVED SOURCES */}
                      {message.role === "assistant" &&
                        message.sources &&
                        message.sources.length > 0 && (
                          <div className="mt-5 border-t border-zinc-800 pt-4">
                            <div className="mb-3 flex items-center gap-2">
                              <BookOpen
                                size={15}
                                className="text-blue-400"
                              />

                              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                                Retrieved sources
                              </h3>

                              <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
                                {message.sources.length}
                              </span>
                            </div>

                            <div className="space-y-3">
                              {message.sources.map(
                                (source, sourceIndex) => (
                                  <div
                                    key={`${source.document_id}-${source.page_number}-${sourceIndex}`}
                                    className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3"
                                  >
                                    <div className="flex items-start gap-3">
                                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
                                        <FileText size={18} />
                                      </div>

                                      <div className="min-w-0 flex-1">
                                        <p
                                          className="break-words text-sm font-medium text-zinc-200"
                                          title={source.filename}
                                        >
                                          {source.filename}
                                        </p>

                                        <div className="mt-1 flex flex-wrap items-center gap-2">
                                          <span className="text-xs text-zinc-500">
                                            Page {source.page_number}
                                          </span>

                                          <span className="text-zinc-700">
                                            ·
                                          </span>

                                          <span className="text-xs text-blue-400">
                                            Retrieved passage
                                          </span>
                                        </div>
                                      </div>
                                    </div>

                                    {source.excerpt && (
                                      <div className="mt-3 rounded-lg bg-zinc-950/70 p-3">
                                        <p className="text-xs leading-6 text-zinc-400">
                                          {source.excerpt}
                                        </p>
                                      </div>
                                    )}
                                  </div>
                                )
                              )}
                            </div>

                            <p className="mt-3 text-xs leading-relaxed text-zinc-600">
                              These passages were retrieved from your
                              uploaded documents and supplied to the AI
                              as reference material. They are not
                              necessarily all explicitly used in the answer.
                            </p>
                          </div>
                        )}
                    </div>
                  </div>
                ))}

                <div ref={bottomRef} />
              </div>
            )}

            {/* REQUEST ERROR */}
            {error && (
              <div className="mt-6 rounded-xl border border-red-900 bg-red-950/30 p-4 text-sm text-red-300">
                <strong>Request failed:</strong> {error}
              </div>
            )}
          </div>
        </section>

        {/* MESSAGE COMPOSER */}
        <footer className="shrink-0 px-4 pb-6 pt-3">
          <div className="mx-auto max-w-3xl">
            <div className="rounded-2xl border border-zinc-700 bg-zinc-900 p-3 shadow-lg focus-within:border-zinc-500">
              {/* HIDDEN PDF INPUT */}
              <input
                ref={chatFileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];

                  if (file) {
                    void attachDocumentToChat(file);
                  }
                }}
              />

              {/* ATTACHED DOCUMENT */}
              {attachedDocument && (
                <div className="mb-2 flex items-center justify-between rounded-lg border border-blue-900/70 bg-blue-950/30 px-3 py-2">
                  <div className="flex min-w-0 items-center gap-2 text-sm text-blue-200">
                    <FileText
                      size={16}
                      className="shrink-0"
                    />

                    <span className="truncate">
                      {attachedDocument.filename}
                    </span>

                    <span className="shrink-0 text-xs text-blue-400">
                      Attached
                    </span>
                  </div>

                  <button
                    type="button"
                    onClick={() => setAttachedDocument(null)}
                    className="ml-2 rounded-md p-1 text-zinc-400 hover:bg-zinc-800 hover:text-white"
                    title="Remove attachment"
                  >
                    <X size={16} />
                  </button>
                </div>
              )}

              {/* TEXT INPUT */}
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Message RoboResearch AI..."
                rows={2}
                disabled={loading}
                className="max-h-40 w-full resize-none bg-transparent px-2 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
              />

              {/* COMPOSER ACTIONS */}
              <div className="flex items-center justify-between px-2 pt-2">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() =>
                      chatFileInputRef.current?.click()
                    }
                    disabled={uploading || loading}
                    className="flex items-center gap-2 rounded-lg px-2 py-1 text-xs text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
                    title="Attach a research paper PDF"
                  >
                    {uploading ? (
                      <LoaderCircle
                        size={16}
                        className="animate-spin"
                      />
                    ) : (
                      <Paperclip size={16} />
                    )}

                    {uploading
                      ? "Uploading PDF..."
                      : "Attach PDF"}
                  </button>

                  <p className="text-xs text-zinc-600">
                    Enter to send · Shift + Enter for a new line
                  </p>
                </div>

                <button
                  onClick={() => void sendMessage()}
                  disabled={
                    (!input.trim() && !attachedDocument) ||
                    loading ||
                    uploading ||
                    !selectedModel
                  }
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
