import { useEffect, useMemo, useState } from "react";
import "./index.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function extractKbIds(answer) {
  if (!answer) return [];

  return [
    ...new Set(
      answer.match(/\[KB-[A-Za-z0-9_-]+\]/g)?.map((id) => id.slice(1)) || []
    ),
  ];
}

function renderAnswer(answer) {
  if (!answer) return null;

  const parts = answer.split(/(\[KB-[A-Za-z0-9_-]+\])/g);

  return parts.map((part, index) =>
    /^\[KB-[A-Za-z0-9_-]+\]$/.test(part) ? (
      <span className="citation-chip" key={`${part}-${index}`}>
        {part}
      </span>
    ) : (
      <span key={index}>{part}</span>
    )
  );
}

function App() {
  const [complaint, setComplaint] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [feedbackState, setFeedbackState] = useState("");
  const [theme, setTheme] = useState(
    () => localStorage.getItem("nexolve-theme") || "light"
  );

  const kbIds = useMemo(() => extractKbIds(answer), [answer]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("nexolve-theme", theme);
  }, [theme]);

  const resolveComplaint = async () => {
    if (!complaint.trim()) {
      setError("Please enter a customer complaint.");
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");
    setFeedback(null);
    setFeedbackState("");

    try {
      const response = await fetch(`${API_BASE_URL}/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ complaint: complaint.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to generate a resolution.");
      }

      setAnswer(data.answer || "");
    } catch (err) {
      setError(
        err.message || "Something went wrong while generating the resolution."
      );
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (value) => {
    if (!kbIds.length) {
      setFeedback(value);
      setFeedbackState(
        "Feedback recorded for this answer. No cited knowledge-base source was available to rerank."
      );
      return;
    }

    setFeedback(value);
    setFeedbackState("Saving feedback…");

    const feedbackValue = value === "positive" ? 1 : -1;

    try {
      const results = await Promise.all(
        kbIds.map((kbId) =>
          fetch(`${API_BASE_URL}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              query: complaint.trim(),
              kb_id: kbId,
              feedback: feedbackValue,
            }),
          }).then(async (response) => {
            if (!response.ok) {
              const data = await response.json().catch(() => ({}));
              throw new Error(data.detail || "Feedback request failed.");
            }
            return response.json();
          })
        )
      );

      if (results.length) {
        setFeedbackState(
          "Thanks — your feedback will influence ranking for similar future queries."
        );
      }
    } catch (err) {
      setFeedbackState(err.message || "Could not save feedback.");
    }
  };

  const toggleTheme = () => {
    setTheme((current) => (current === "light" ? "dark" : "light"));
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">N</div>
          <div>
            <h1>Nexolve</h1>
            <span>Resolution Copilot</span>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="status">
            <span className="status-dot"></span>
            AI Resolution Assistant
          </div>
          <button className="theme-toggle" onClick={toggleTheme} type="button">
            <span>{theme === "light" ? "☾" : "☀"}</span>
            {theme === "light" ? "Dark" : "Light"}
          </button>
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <p className="eyebrow">CUSTOMER SUPPORT INTELLIGENCE</p>
          <div className="hero-row">
            <div>
              <h2>
                Resolve customer issues
                <br />
                with grounded answers.
              </h2>
              <p className="hero-description">
                Enter a customer complaint and generate a knowledge-base
                grounded resolution for the support agent.
              </p>
            </div>
            <div className="hero-stat">
              <span className="stat-dot"></span>
              <div>
                <strong>Evidence grounded</strong>
                <small>Retrieval + guarded generation</small>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="input-card card">
            <div className="card-header">
              <div>
                <div className="section-label">01</div>
                <h3>Customer Complaint</h3>
                <p>Describe the customer's issue in natural language.</p>
              </div>
            </div>

            <textarea
              value={complaint}
              onChange={(event) => setComplaint(event.target.value)}
              placeholder="Example: My internet keeps disconnecting every evening and video calls keep dropping..."
              disabled={loading}
              onKeyDown={(event) => {
                if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                  resolveComplaint();
                }
              }}
            />

            <div className="input-hint">Ctrl + Enter to resolve</div>

            {error && <div className="error">{error}</div>}

            <button
              className="resolve-button"
              onClick={resolveComplaint}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Generating resolution...
                </>
              ) : (
                <>
                  Resolve Issue <span>→</span>
                </>
              )}
            </button>
          </div>

          <div className="result-card card">
            <div className="card-header">
              <div>
                <div className="section-label">02</div>
                <h3>Resolution</h3>
                <p>AI-generated response grounded in the knowledge base.</p>
              </div>
              {answer && <span className="grounded-badge">GROUNDED</span>}
            </div>

            {!answer && !loading && (
              <div className="empty-state">
                <div className="empty-icon">✦</div>
                <h4>No resolution yet</h4>
                <p>Enter a complaint on the left to generate a resolution.</p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <h4>Analyzing complaint...</h4>
                <p>Searching the knowledge base and generating a grounded resolution.</p>
              </div>
            )}

            {answer && !loading && (
              <>
                <div className="answer-box">{renderAnswer(answer)}</div>

                <div className="source-row">
                  <div>
                    <span className="source-label">Sources used</span>
                    <div className="source-list">
                      {kbIds.length ? (
                        kbIds.map((id) => (
                          <span className="source-chip" key={id}>{id}</span>
                        ))
                      ) : (
                        <span className="source-muted">No KB citation detected</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="feedback">
                  <div>
                    <strong>Was this resolution useful?</strong>
                    <span>
                      Your rating is matched to the cited sources for this query and
                      used by the feedback reranker.
                    </span>
                  </div>
                  <div className="feedback-buttons">
                    <button
                      aria-label="Helpful resolution"
                      className={feedback === "positive" ? "feedback-active positive" : ""}
                      onClick={() => sendFeedback("positive")}
                      type="button"
                    >
                      👍
                    </button>
                    <button
                      aria-label="Unhelpful resolution"
                      className={feedback === "negative" ? "feedback-active negative" : ""}
                      onClick={() => sendFeedback("negative")}
                      type="button"
                    >
                      👎
                    </button>
                  </div>
                </div>

                {feedbackState && <div className="feedback-state">{feedbackState}</div>}
              </>
            )}
          </div>
        </section>
      </main>

      <footer>
        <span>Nexolve Resolution Copilot</span>
        <span>Knowledge-grounded AI support · Feedback-aware retrieval</span>
      </footer>
    </div>
  );
}

export default App;
