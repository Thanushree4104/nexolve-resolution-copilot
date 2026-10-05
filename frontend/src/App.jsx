import { useState } from "react";
import "./index.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [complaint, setComplaint] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState(null);

  const resolveComplaint = async () => {
    if (!complaint.trim()) {
      setError("Please enter a customer complaint.");
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");
    setFeedback(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/resolve`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            complaint: complaint.trim(),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to generate a resolution."
        );
      }

      setAnswer(data.answer || "");
    } catch (err) {
      setError(
        err.message ||
          "Something went wrong while generating the resolution."
      );
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (positive) => {
    /*
     * Feedback endpoint will be connected after the
     * frontend is working with /resolve.
     *
     * For now we only update the UI.
     */
    setFeedback(positive ? "positive" : "negative");
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

        <div className="status">
          <span className="status-dot"></span>
          AI Resolution Assistant
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div>
            <p className="eyebrow">
              CUSTOMER SUPPORT INTELLIGENCE
            </p>

            <h2>
              Resolve customer issues
              <br />
              with grounded answers.
            </h2>

            <p className="hero-description">
              Enter a customer complaint and generate a
              knowledge-base grounded resolution for the
              support agent.
            </p>
          </div>
        </section>

        <section className="workspace">
          <div className="input-card card">
            <div className="card-header">
              <div>
                <h3>Customer Complaint</h3>
                <p>
                  Describe the customer's issue in natural
                  language.
                </p>
              </div>
            </div>

            <textarea
              value={complaint}
              onChange={(event) =>
                setComplaint(event.target.value)
              }
              placeholder="Example: My internet keeps disconnecting every evening and video calls keep dropping..."
              disabled={loading}
            />

            {error && (
              <div className="error">
                {error}
              </div>
            )}

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
                  Resolve Issue
                  <span>→</span>
                </>
              )}
            </button>
          </div>

          <div className="result-card card">
            <div className="card-header">
              <div>
                <h3>Resolution</h3>
                <p>
                  AI-generated response grounded in the
                  knowledge base.
                </p>
              </div>

              {answer && (
                <span className="grounded-badge">
                  GROUNDED
                </span>
              )}
            </div>

            {!answer && !loading && (
              <div className="empty-state">
                <div className="empty-icon">✦</div>

                <h4>No resolution yet</h4>

                <p>
                  Enter a complaint on the left to generate
                  a resolution.
                </p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>

                <h4>Analyzing complaint...</h4>

                <p>
                  Searching the knowledge base and
                  generating a grounded resolution.
                </p>
              </div>
            )}

            {answer && !loading && (
              <>
                <div className="answer-box">
                  {answer}
                </div>

                <div className="feedback">
                  <span>
                    Was this resolution useful?
                  </span>

                  <div className="feedback-buttons">
                    <button
                      className={
                        feedback === "positive"
                          ? "feedback-active"
                          : ""
                      }
                      onClick={() =>
                        sendFeedback(true)
                      }
                    >
                      👍
                    </button>

                    <button
                      className={
                        feedback === "negative"
                          ? "feedback-active"
                          : ""
                      }
                      onClick={() =>
                        sendFeedback(false)
                      }
                    >
                      👎
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </section>
      </main>

      <footer>
        <span>Nexolve Resolution Copilot</span>
        <span>Knowledge-grounded AI support</span>
      </footer>
    </div>
  );
}

export default App;