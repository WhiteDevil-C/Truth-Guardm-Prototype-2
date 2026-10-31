// TruthGuard popup logic.
// Talks to the local FastAPI backend at API_BASE and renders the full
// response: label, confidence, class probabilities, XAI explanation,
// warnings, and manual "check further" search links.

const API_BASE = "http://127.0.0.1:8000";
const ANALYZE_URL = `${API_BASE}/analyze`;
const EVIDENCE_URL = `${API_BASE}/evidence`;
const HEALTH_URL = `${API_BASE}/`;

const MAX_CHARS = 20000; // keeps huge pages reasonably sized before sending

// --- Element references ---------------------------------------------------
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const errorTitle = document.getElementById("errorTitle");
const errorMessage = document.getElementById("errorMessage");
const result = document.getElementById("result");

const labelBadge = document.getElementById("labelBadge");
const labelDescription = document.getElementById("labelDescription");
const inputWarning = document.getElementById("inputWarning");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBarFill = document.getElementById("confidenceBarFill");
const probList = document.getElementById("probList");
const explainNote = document.getElementById("explainNote");
const positiveClues = document.getElementById("positiveClues");
const negativeClues = document.getElementById("negativeClues");
const evidenceList = document.getElementById("evidenceList");
const cautionText = document.getElementById("cautionText");

const analyzePageBtn = document.getElementById("analyzePage");
const checkSelectedTextBtn = document.getElementById("checkSelectedText");

// --- View state helpers ------------------------------------------------- --

function showView(view) {
    emptyState.classList.toggle("hidden", view !== "empty");
    loadingState.classList.toggle("hidden", view !== "loading");
    errorState.classList.toggle("hidden", view !== "error");
    result.classList.toggle("hidden", view !== "result");
}

function showError(title, message) {
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    showView("error");
}

function setButtonsDisabled(disabled) {
    analyzePageBtn.disabled = disabled;
    checkSelectedTextBtn.disabled = disabled;
}

// --- Backend status indicator --------------------------------------------

async function refreshBackendStatus() {
    statusDot.className = "status-dot checking";
    statusText.textContent = "Checking backend…";
    try {
        const response = await fetch(HEALTH_URL, { method: "GET" });
        if (!response.ok) throw new Error("bad status");
        const data = await response.json();
        if (data.model_loaded) {
            statusDot.className = "status-dot online";
            statusText.textContent = "Backend connected · model loaded";
        } else {
            statusDot.className = "status-dot offline";
            statusText.textContent = "Backend connected, but model failed to load";
        }
    } catch (error) {
        statusDot.className = "status-dot offline";
        statusText.textContent = "Backend unavailable (is uvicorn running on :8000?)";
    }
}

// --- Page text extraction -------------------------------------------------

async function getTextFromTab(mode) {
    const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    if (!tab || !tab.id) {
        throw new Error("No active tab found.");
    }

    // Retrieve the selection saved by content.js before the popup opened.
    if (mode === "selected") {
        const saved = await chrome.storage.local.get("truthguardSelection");
        const selection = saved.truthguardSelection;

        if (!selection || !selection.text) {
            throw new Error("No saved text selection found. Select text on the webpage first.");
        }

        // Prevent using a selection saved from another webpage.
        if (selection.url !== tab.url) {
            throw new Error("The saved selection belongs to another webpage. Select the text again.");
        }

        return selection.text;
    }

    // Existing page extraction logic.
    const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
            // Avoid treating the Google News homepage as one article.
            if (
                location.hostname === "news.google.com" &&
                (location.pathname === "/home" || location.pathname === "/")
            ) {
                return "__TRUTHGUARD_GOOGLE_NEWS_HOME__";
            }

            // Work on a copy so the original webpage is not changed.
            const clone = document.body.cloneNode(true);

            clone.querySelectorAll(
                "script, style, noscript, nav, header, footer, aside, " +
                "form, button, input, select, textarea, iframe, " +
                "[role='navigation'], [role='banner'], " +
                "[role='contentinfo'], [aria-hidden='true']"
            ).forEach(element => element.remove());

            // Prefer the actual article body where available.
            const content =
                clone.querySelector('[itemprop="articleBody"]') ||
                clone.querySelector("article") ||
                clone.querySelector("main, [role='main']") ||
                clone;

            // Extract readable paragraphs and discard empty ones.
            const paragraphs = [...content.querySelectorAll("p")]
                .map(paragraph => paragraph.innerText.trim())
                .filter(text => text.length > 0);

            if (paragraphs.length > 0) {
                return paragraphs.join("\n\n");
            }

            return content.innerText || content.textContent || "";
        }
    });

    return results[0] && results[0].result;
}

// --- Rendering --------------------------------------------------------------

function renderLabel(data) {
    const label = data.label || "uncertain";
    labelBadge.textContent = label.replace(/-/g, " ");
    labelBadge.className = `label-badge ${label}`;
    labelDescription.textContent = data.label_description || "";
}

function renderWarning(data) {
    if (data.input_warning) {
        inputWarning.textContent = `⚠️ ${data.input_warning}`;
        inputWarning.classList.remove("hidden");
    } else {
        inputWarning.classList.add("hidden");
    }
}

function renderConfidence(data) {
    const pct = Math.round((data.confidence || 0) * 100);
    confidenceValue.textContent = `${pct}%`;
    confidenceBarFill.style.width = `${pct}%`;
}

function renderProbabilities(data) {
    probList.innerHTML = "";
    const entries = Object.entries(data.class_probabilities || {});
    if (entries.length === 0) {
        probList.innerHTML = '<p class="muted-note">No probability distribution available.</p>';
        return;
    }
    entries
        .sort((a, b) => b[1] - a[1])
        .forEach(([label, prob]) => {
            const pct = Math.round(prob * 100);
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-row-top">
                    <span>${label.replace(/-/g, " ")}</span>
                    <span>${pct}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill ${label === data.label ? "active" : ""}" style="width:${pct}%"></div>
                </div>`;
            probList.appendChild(row);
        });
}

function renderClueList(container, clues) {
    container.replaceChildren();

    if (!Array.isArray(clues) || clues.length === 0) {
        const note = document.createElement("li");
        note.textContent = "No influential terms were identified.";
        note.className = "muted-note";
        container.appendChild(note);
        return;
    }

    const validClues = clues
        .filter((clue) =>
            clue &&
            typeof clue.term === "string" &&
            Number.isFinite(Number(clue.contribution))
        )
        .slice(0, 5);

    if (validClues.length === 0) {
        const note = document.createElement("li");
        note.textContent = "Explanation details are unavailable.";
        note.className = "muted-note";
        container.appendChild(note);
        return;
    }

    const maxMagnitude = Math.max(
        ...validClues.map((clue) => Math.abs(Number(clue.contribution))),
        0.000001
    );

    validClues.forEach((clue) => {
        const value = Number(clue.contribution);
        const magnitude = Math.abs(value);
        const barWidth = Math.max(3, (magnitude / maxMagnitude) * 100);

        const item = document.createElement("li");
        item.className = "xai-clue";

        const top = document.createElement("div");
        top.className = "xai-clue-top";

        const term = document.createElement("span");
        term.className = "clue-term";
        term.textContent = clue.term;

        const score = document.createElement("span");
        score.className = "clue-value " +
            (value >= 0 ? "xai-positive" : "xai-negative");
        score.textContent = (value > 0 ? "+" : "") + value.toFixed(4);

        top.append(term, score);

        const track = document.createElement("div");
        track.className = "xai-bar-track";
        track.setAttribute(
            "aria-label",
            "Relative contribution magnitude " + Math.round(barWidth) + " percent"
        );

        const fill = document.createElement("div");
        fill.className = "xai-bar-fill " +
            (value >= 0 ? "xai-positive-fill" : "xai-negative-fill");
        fill.style.width = barWidth + "%";

        track.appendChild(fill);
        item.append(top, track);
        container.appendChild(item);
    });
}

function renderExplanation(data) {
    const explanation = data.explanation || {};
    explainNote.textContent = explanation.note || "";
    renderClueList(positiveClues, explanation.positive_clues);
    renderClueList(negativeClues, explanation.negative_clues);
}

function renderCaution(data) {
    cautionText.textContent = data.caution || "";
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function renderResult(data) {
    renderLabel(data);
    renderWarning(data);
    renderConfidence(data);
    renderProbabilities(data);
    renderExplanation(data);
    renderCaution(data);
    showView("result");
}

// --- Evidence / Google fact-check reviews and manual search links -------------

async function loadEvidence(text) {
    evidenceList.innerHTML =
        '<p class="muted-note">Searching published fact-check reviews…</p>';

    try {
        const response = await fetch(EVIDENCE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text.slice(0, 2000) }),
        });

        if (!response.ok) throw new Error("Evidence request failed");

        const data = await response.json();
        evidenceList.innerHTML = "";

        const heading = document.createElement("h3");
        heading.className = "evidence-heading";
        heading.textContent = "Published fact-check reviews";
        evidenceList.appendChild(heading);

        if (data.status === "results" && Array.isArray(data.fact_checks)) {
            data.fact_checks.forEach((claim, index) => {
                const card = document.createElement("article");
                card.className = "factcheck-card";

                const matchTitle = document.createElement("h4");
                matchTitle.textContent = `Match ${index + 1}`;
                card.appendChild(matchTitle);

                if (claim.claim_text) {
                    const claimText = document.createElement("p");
                    claimText.className = "factcheck-claim";
                    claimText.textContent = claim.claim_text;
                    card.appendChild(claimText);
                }

                if (claim.claimant) {
                    const claimant = document.createElement("p");
                    claimant.className = "factcheck-meta";
                    claimant.textContent = `Claimant: ${claim.claimant}`;
                    card.appendChild(claimant);
                }

                if (Array.isArray(claim.reviews) && claim.reviews.length) {
                    claim.reviews.forEach((review) => {
                        const reviewBox = document.createElement("div");
                        reviewBox.className = "factcheck-review";

                        const publisher = document.createElement("p");
                        publisher.className = "factcheck-publisher";
                        publisher.textContent =
                            review.publisher || "Unknown publisher";
                        reviewBox.appendChild(publisher);

                        if (review.title) {
                            const title = document.createElement("p");
                            title.className = "factcheck-title";
                            title.textContent = review.title;
                            reviewBox.appendChild(title);
                        }

                        const rating = document.createElement("p");
                        rating.className = "factcheck-rating";
                        rating.textContent =
                            `Published rating: ${review.rating || "Not supplied"}`;
                        reviewBox.appendChild(rating);

                        if (review.review_date) {
                            const date = document.createElement("p");
                            date.className = "factcheck-meta";
                            date.textContent = `Review date: ${review.review_date}`;
                            reviewBox.appendChild(date);
                        }

                        if (review.url) {
                            try {
                                const parsedUrl = new URL(review.url);
                                if (parsedUrl.protocol === "https:" ||
                                    parsedUrl.protocol === "http:") {
                                    const link = document.createElement("a");
                                    link.href = parsedUrl.href;
                                    link.target = "_blank";
                                    link.rel = "noopener noreferrer";
                                    link.className = "evidence-link";
                                    link.textContent = "Read original fact-check ↗";
                                    reviewBox.appendChild(link);
                                }
                            } catch (_) {
                                // Ignore malformed source URLs.
                            }
                        }

                        card.appendChild(reviewBox);
                    });
                } else {
                    const noReviews = document.createElement("p");
                    noReviews.className = "muted-note";
                    noReviews.textContent =
                        "No individual reviews were included in this match.";
                    card.appendChild(noReviews);
                }

                evidenceList.appendChild(card);
            });

            const contextNote = document.createElement("p");
            contextNote.className = "muted-note evidence-notice";
            contextNote.textContent =
                data.note ||
                "Review the original sources carefully. A matching claim may have a different context.";
            evidenceList.appendChild(contextNote);
        } else {
            const message = document.createElement("p");
            message.className = "muted-note";
            message.textContent =
                data.status === "no_matches"
                    ? "Google found no matching published fact-check reviews. This does not mean the claim is true or false."
                    : data.note || "Published fact-check results are currently unavailable.";
            evidenceList.appendChild(message);
        }

        const sources = Array.isArray(data.sources) ? data.sources : [];
        if (sources.length) {
            const manualHeading = document.createElement("h3");
            manualHeading.className = "evidence-heading";
            manualHeading.textContent = "Search manually";
            evidenceList.appendChild(manualHeading);

            const manualNote = document.createElement("p");
            manualNote.className = "muted-note";
            manualNote.textContent =
                "These are search shortcuts, not confirmed fact-check results.";
            evidenceList.appendChild(manualNote);

            sources.forEach((source) => {
                if (!source.url) return;

                try {
                    const parsedUrl = new URL(source.url);
                    if (parsedUrl.protocol !== "https:" &&
                        parsedUrl.protocol !== "http:") return;

                    const link = document.createElement("a");
                    link.href = parsedUrl.href;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.className = "evidence-link";
                    link.textContent = `🔗 ${source.name || "Search source"}`;
                    evidenceList.appendChild(link);
                } catch (_) {
                    // Ignore malformed source URLs.
                }
            });
        }
    } catch (error) {
        console.error("TruthGuard evidence lookup failed:", error);
        evidenceList.innerHTML =
            '<p class="muted-note">Could not load evidence. Check that the TruthGuard backend is running.</p>';
    }
}

// --- Collapsible sections ---------------------------------------------------

document.querySelectorAll(".section-toggle").forEach((toggle) => {
    toggle.addEventListener("click", () => {
        const target = document.getElementById(toggle.dataset.target);
        const isOpen = target.classList.toggle("open");
        toggle.classList.toggle("open", isOpen);
    });
});

// Open the probabilities section by default once a result is shown.
function openDefaultSections() {
    const probSection = document.getElementById("probSection");
    const probToggle = document.querySelector('[data-target="probSection"]');
    probSection.classList.add("open");
    probToggle.classList.add("open");
}

// --- Main analyze flow -------------------------------------------------------

async function analyze(mode) {
    let text;

    setButtonsDisabled(true);
    showView("loading");

    try {
        text = await getTextFromTab(mode);
    } catch (error) {
        console.error(error);
        setButtonsDisabled(false);
        showError(
            "Can't read this page",
            "TruthGuard can't access this tab (try a normal http/https webpage, not a Chrome internal page)."
        );
        return;
    }

    if (text === "__TRUTHGUARD_GOOGLE_NEWS_HOME__") {
        setButtonsDisabled(false);
        showError(
            "Select an individual news story",
            "Google News contains multiple headlines. Select a specific headline or story on the page, then click Check Selected Text."
        );
        return;
    }

    if (!text || text.trim().length === 0) {
        setButtonsDisabled(false);
        showError(
            mode === "selected" ? "No text selected" : "No readable text",
            mode === "selected"
                ? "Select some text on the webpage first, then click \"Check Selected Text\" again."
                : "This page does not appear to contain readable text."
        );
        return;
    }

    text = text.trim().slice(0, MAX_CHARS);

    try {
        const response = await fetch(ANALYZE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            let detail = `HTTP ${response.status}`;
            try {
                const body = await response.json();
                if (body && body.detail) detail = JSON.stringify(body.detail);
            } catch (_) {
                /* ignore parse errors */
            }
            throw new Error(detail);
        }

        const data = await response.json();
        renderResult(data);
        openDefaultSections();
        loadEvidence(text);
        refreshBackendStatus();
    } catch (error) {
        console.error(error);
        showError(
            "Backend request failed",
            "Could not reach the TruthGuard backend. Make sure uvicorn is running on http://127.0.0.1:8000."
        );
        refreshBackendStatus();
    } finally {
        setButtonsDisabled(false);
    }
}

analyzePageBtn.addEventListener("click", () => analyze("page"));
checkSelectedTextBtn.addEventListener("click", () => analyze("selected"));

// Initial state
showView("empty");
refreshBackendStatus();
