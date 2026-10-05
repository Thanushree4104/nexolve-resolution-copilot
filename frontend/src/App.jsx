import { useEffect, useMemo, useState } from "react";
import "./index.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [complaint, setComplaint] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [feedback, setFeedback] = useState(null);
  const [feedbackMessage, setFeedbackMessage] = useState("");

  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("theme") === "dark";
  });

  // ------------------------------------------------------------
  // THEME
  // ------------------------------------------------------------

  useEffect(() => {
    document.documentElement.setAttribute(
      "data-theme",
      darkMode ? "dark" : "light"
    );

    localStorage.setItem(
      "theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  // ------------------------------------------------------------
  // RESOLUTION
  // ------------------------------------------------------------

  const handleResolve = async () => {
    if (!complaint.trim() || loading) return;

    setLoading(true);
    setResult(null);
    setFeedback(null);
    setFeedbackMessage("");

    try {
      const response = await fetch(`${API_BASE}/resolve`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          complaint: complaint.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error("Resolution error:", error);

      setResult({
        answer:
          "We couldn't generate a resolution right now. Please try again in a moment.",
        error: true,
      });
    } finally {
      setLoading(false);
    }
  };

  // ------------------------------------------------------------
  // FEEDBACK
  // ------------------------------------------------------------

  const handleFeedback = async (value) => {
    if (!result || feedback !== null) return;

    setFeedback(value);
    setFeedbackMessage("");

    try {
      const response = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: complaint.trim(),
          answer: result.answer || "",
          feedback: value,
          retrieved_ids:
            result.retrieved_ids ||
            result.sources ||
            [],
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Feedback request failed: ${response.status}`
        );
      }

      setFeedbackMessage(
        "Thanks for your feedback. It will help improve future resolutions."
      );
    } catch (error) {
      console.error("Feedback error:", error);

      /*
       * Feedback is intentionally kept as a lightweight
       * user interaction. Even if the optional endpoint
       * fails, don't make the user interact again.
       */
      setFeedbackMessage(
        "Thanks for your feedback."
      );
    }
  };

  // ------------------------------------------------------------
  // EVIDENCE STATUS
  // ------------------------------------------------------------

  const evidenceStatus = useMemo(() => {
    const answer = (result?.answer || "").toLowerCase();

    const insufficientEvidence =
      answer.includes(
        "available knowledge-base evidence is insufficient"
      ) ||
      answer.includes(
        "knowledge-base evidence is insufficient"
      ) ||
      answer.includes(
        "does not support a safe"
      ) ||
      answer.includes(
        "no relevant knowledge"
      ) ||
      answer.includes(
        "not enough evidence"
      );

    if (insufficientEvidence) {
      return {
        type: "limited",
        label: "LIMITED EVIDENCE",
        icon: "!",
      };
    }

    return {
      type: "grounded",
      label: "GROUNDED",
      icon: "✓",
    };
  }, [result]);

  // ------------------------------------------------------------
  // SOURCES
  // ------------------------------------------------------------

  const sources = useMemo(() => {
    if (!result) return [];

    const rawSources =
      result.sources ||
      result.retrieved_ids ||
      [];

    if (!Array.isArray(rawSources)) {
      return [];
    }

    /*
     * Remove duplicates while preserving order.
     */
    return [...new Set(rawSources)].filter(Boolean);
  }, [result]);

  // ------------------------------------------------------------
  // ANSWER FORMATTER
  // ------------------------------------------------------------

  const formatAnswer = (answer) => {
    if (!answer) return null;

    const lines = answer.split("\n");

    return lines.map((line, index) => {
      const trimmed = line.trim();

      if (!trimmed) {
        return (
          <div
            className="answer-spacer"
            key={index}
          />
        );
      }

      /*
       * Section headings such as:
       *
       * Likely issue:
       * Recommended troubleshooting steps:
       * Escalation condition:
       */
      if (
        trimmed.endsWith(":") &&
        trimmed.length < 100
      ) {
        return (
          <h3
            className="answer-heading"
            key={index}
          >
            {trimmed.replace(/:$/, "")}
          </h3>
        );
      }

      /*
       * Numbered troubleshooting steps.
       */
      if (/^\d+[\.\)]\s/.test(trimmed)) {
        const match = trimmed.match(
          /^(\d+[\.\)])\s+(.*)$/
        );

        return (
          <div
            className="answer-step"
            key={index}
          >
            <span className="step-number">
              {match?.[1]}
            </span>

            <span>
              {renderCitations(
                match?.[2] || trimmed
              )}
            </span>
          </div>
        );
      }

      return (
        <p
          className="answer-paragraph"
          key={index}
        >
          {renderCitations(trimmed)}
        </p>
      );
    });
  };

  const renderCitations = (text) => {
    const parts = text.split(
      /(\[KB-[A-Za-z0-9_-]+\])/g
    );

    return parts.map((part, index) => {
      if (
        /^\[KB-[A-Za-z0-9_-]+\]$/.test(part)
      ) {
        return (
          <span
            className="citation"
            key={index}
          >
            {part}
          </span>
        );
      }

      return (
        <span key={index}>
          {part}
        </span>
      );
    });
  };

  // ------------------------------------------------------------
  // KEYBOARD SHORTCUT
  // ------------------------------------------------------------

  const handleKeyDown = (event) => {
    /*
     * Ctrl + Enter / Cmd + Enter resolves.
     */
    if (
      (event.ctrlKey || event.metaKey) &&
      event.key === "Enter"
    ) {
      event.preventDefault();
      handleResolve();
    }
  };

  // ------------------------------------------------------------
  // UI
  // ------------------------------------------------------------

  return (
    <div className="app">
      {/* ====================================================== */}
      {/* HEADER */}
      {/* ====================================================== */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            N
          </div>

          <div>
            <div className="brand-name">
              Nexolve
            </div>

            <div className="brand-subtitle">
              Resolution Copilot
            </div>
          </div>
        </div>

        <div className="header-actions">
          <div className="status-pill">
            <span className="status-dot" />
            System Online
          </div>

          <button
            className="theme-toggle"
            onClick={() =>
              setDarkMode(
                (previous) => !previous
              )
            }
            aria-label={
              darkMode
                ? "Switch to light mode"
                : "Switch to dark mode"
            }
            title={
              darkMode
                ? "Light mode"
                : "Dark mode"
            }
          >
            {darkMode ? "☀" : "☾"}
          </button>
        </div>
      </header>

      {/* ====================================================== */}
      {/* MAIN */}
      {/* ====================================================== */}

      <main className="main-container">

        {/* HERO */}

        <section className="hero">
          <div className="eyebrow">
            CUSTOMER SUPPORT INTELLIGENCE
          </div>

          <h1>
            Resolve customer issues
            <span> faster.</span>
          </h1>

          <p>
            Turn customer complaints into clear,
            knowledge-grounded support guidance.
          </p>
        </section>

        {/* ================================================== */}
        {/* COMPLAINT INPUT */}
        {/* ================================================== */}

        <section className="complaint-card">
          <div className="section-label">
            CUSTOMER COMPLAINT
          </div>

          <textarea
            value={complaint}
            onChange={(event) =>
              setComplaint(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Describe the customer's issue..."
            rows={6}
            disabled={loading}
          />

          <div className="input-footer">
            <div className="input-hint">
              <span className="character-count">
                {complaint.length} characters
              </span>

              <span className="keyboard-hint">
                Ctrl + Enter to resolve
              </span>
            </div>

            <button
              className="resolve-button"
              onClick={handleResolve}
              disabled={
                loading ||
                !complaint.trim()
              }
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analyzing...
                </>
              ) : (
                <>
                  Resolve Issue
                  <span className="arrow">
                    →
                  </span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* ================================================== */}
        {/* LOADING */}
        {/* ================================================== */}

        {loading && (
          <section className="loading-card">
            <div className="loading-animation">
              <span />
              <span />
              <span />
            </div>

            <h3>
              Analyzing the complaint
            </h3>

            <p>
              Searching relevant support knowledge
              and preparing guidance...
            </p>
          </section>
        )}

        {/* ================================================== */}
        {/* RESULT */}
        {/* ================================================== */}

        {result && !loading && (
          <section className="result-section">

            {/* RESULT HEADER */}

            <div className="result-header">
              <div>
                <div className="section-label">
                  RESOLUTION
                </div>

                <h2>
                  Recommended Resolution
                </h2>

                <p className="result-subtitle">
                  Guidance generated from the
                  available support knowledge.
                </p>
              </div>

              <div
                className={`grounded-badge ${
                  evidenceStatus.type ===
                  "limited"
                    ? "limited"
                    : ""
                }`}
              >
                <span>
                  {evidenceStatus.icon}
                </span>

                {evidenceStatus.label}
              </div>
            </div>

            {/* RESOLUTION */}

            <div
              className={`resolution-card ${
                evidenceStatus.type ===
                "limited"
                  ? "limited-evidence"
                  : ""
              }`}
            >
              <div className="resolution-content">
                {formatAnswer(
                  result.answer
                )}
              </div>
            </div>

            {/* ================================================= */}
            {/* SOURCES */}
            {/* ================================================= */}

            {sources.length > 0 && (
              <div className="sources-card">
                <div className="sources-header">
                  <div>
                    <div className="sources-title">
                      Knowledge sources
                    </div>

                    <div className="sources-subtitle">
                      Support articles used by the
                      resolution engine.
                    </div>
                  </div>

                  <span className="source-count">
                    {sources.length}
                  </span>
                </div>

                <div className="source-list">
                  {sources.map(
                    (source, index) => (
                      <div
                        className="source-chip"
                        key={`${source}-${index}`}
                      >
                        <span className="source-icon">
                          ◈
                        </span>

                        <span>
                          {source}
                        </span>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {/* ================================================= */}
            {/* FEEDBACK */}
            {/* ================================================= */}

            <div className="feedback-card">
              <div className="feedback-text">
                <strong>
                  Was this resolution helpful?
                </strong>

                <span>
                  Your feedback helps improve
                  future recommendations.
                </span>
              </div>

              <div className="feedback-actions">
                <button
                  className={`feedback-button ${
                    feedback === "positive"
                      ? "selected"
                      : ""
                  }`}
                  onClick={() =>
                    handleFeedback(
                      "positive"
                    )
                  }
                  disabled={
                    feedback !== null
                  }
                >
                  <span>👍</span>
                  Helpful
                </button>

                <button
                  className={`feedback-button ${
                    feedback === "negative"
                      ? "selected negative"
                      : ""
                  }`}
                  onClick={() =>
                    handleFeedback(
                      "negative"
                    )
                  }
                  disabled={
                    feedback !== null
                  }
                >
                  <span>👎</span>
                  Not helpful
                </button>
              </div>

              {feedbackMessage && (
                <div className="feedback-confirmation">
                  ✓ {feedbackMessage}
                </div>
              )}
            </div>
          </section>
        )}

        {/* ================================================== */}
        {/* EMPTY STATE */}
        {/* ================================================== */}

        {!result && !loading && (
          <section className="empty-state">
            <div className="empty-icon">
              ✦
            </div>

            <h3>
              Ready to resolve an issue
            </h3>

            <p>
              Enter a customer complaint above
              to generate a recommended
              resolution.
            </p>
          </section>
        )}
      </main>

      {/* ====================================================== */}
      {/* FOOTER */}
      {/* ====================================================== */}

      <footer className="footer">
        <span>
          Nexolve Resolution Copilot
        </span>

        <span className="footer-separator">
          •
        </span>

        <span>
          Knowledge-grounded support
        </span>
      </footer>
    </div>
  );
}

export default App;