const $ = (id) => document.getElementById(id);

function setText(id, value, fallback = "Not supplied") {
    $(id).textContent = value == null || value === "" ? fallback : String(value);
}

function safeUrl(value) {
    try {
        const url = new URL(value);
        return ["https:", "http:"].includes(url.protocol) ? url.href : null;
    } catch (_) {
        return null;
    }
}

function addLink(parent, label, value, className) {
    const url = safeUrl(value);
    if (!url) return;

    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.className = className;
    link.textContent = label;
    parent.appendChild(link);
}

function renderProbabilities(data) {
    const root = $("probabilities");
    root.replaceChildren();

    const probs = data.class_probabilities || data.probabilities || {};
    const entries = Array.isArray(probs)
        ? probs.map((item) => [item.label || item.class || item.name, item.probability ?? item.score ?? item.value])
        : Object.entries(probs);

    if (!entries.length) {
        root.textContent = "Class probability information is unavailable.";
        return;
    }

    entries.forEach(([label, raw]) => {
        const value = Number(raw);
        const valid = Number.isFinite(value);
        const percent = valid ? Math.max(0, Math.min(100, value * 100)) : 0;

        const row = document.createElement("div");
        row.className = "prob-row";

        const head = document.createElement("div");
        head.className = "prob-head";

        const name = document.createElement("strong");
        name.textContent = String(label);

        const number = document.createElement("span");
        number.textContent = valid ? `${percent.toFixed(2)}%` : "Unavailable";

        head.append(name, number);

        const bar = document.createElement("div");
        bar.className = "bar";

        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = `${percent}%`;

        bar.appendChild(fill);
        row.append(head, bar);
        root.appendChild(row);
    });
}

function renderClues(id, clues) {
    const root = $(id);
    root.replaceChildren();

    if (!Array.isArray(clues) || !clues.length) {
        root.textContent = "No explanation clues supplied.";
        return;
    }

    clues.forEach((item) => {
        const box = document.createElement("div");
        box.className = "clue";

        if (typeof item === "string") {
            box.textContent = item;
        } else {
            const term = item.term || item.feature || item.text || "Feature";
            const contribution = item.contribution ?? item.weight;
            box.textContent = contribution == null
                ? String(term)
                : `${term} — contribution: ${contribution}`;
        }

        root.appendChild(box);
    });
}

function renderEvidence(saved) {
    const status = $("evidenceStatus");
    const checks = $("factChecks");
    const sourcesRoot = $("manualSources");
    checks.replaceChildren();
    sourcesRoot.replaceChildren();

    if (saved.evidenceStatus === "pending" || !saved.evidenceStatus) {
        status.textContent = "Evidence lookup is still running. This report will update when it finishes.";
        return;
    }

    if (saved.evidenceStatus === "unavailable") {
        status.textContent = "Evidence lookup could not be completed. Check backend connectivity and reopen or refresh this report after another analysis.";
        return;
    }

    const data = saved.evidence;
    if (!data) {
        status.textContent = "No evidence response was saved.";
        return;
    }

    status.textContent = data.note || (
        data.status === "no_matches"
            ? "No matching published fact-check reviews were found. This does not mean the claim is true or false."
            : "Evidence lookup completed."
    );

    if (Array.isArray(data.fact_checks)) {
        data.fact_checks.forEach((claim, index) => {
            const card = document.createElement("article");
            card.className = "fact-card";

            const heading = document.createElement("h3");
            heading.textContent = `Fact-check match ${index + 1}`;
            card.appendChild(heading);

            if (claim.claim_text) {
                const claimText = document.createElement("p");
                claimText.textContent = claim.claim_text;
                card.appendChild(claimText);
            }

            if (claim.claimant) {
                const claimant = document.createElement("p");
                claimant.textContent = `Claimant: ${claim.claimant}`;
                card.appendChild(claimant);
            }

            if (Array.isArray(claim.reviews)) {
                claim.reviews.forEach((review) => {
                    const publisher = document.createElement("p");
                    publisher.className = "fact-publisher";
                    publisher.textContent = review.publisher || "Unknown publisher";

                    const title = document.createElement("p");
                    title.textContent = review.title || "Untitled review";

                    const rating = document.createElement("p");
                    rating.textContent = `Published rating: ${review.rating || "Not supplied"}`;

                    const date = document.createElement("p");
                    date.textContent = review.review_date
                        ? `Review date: ${review.review_date}`
                        : "";

                    card.append(publisher, title, rating, date);
                    addLink(card, "Read original fact-check ↗", review.url, "fact-link");
                });
            }

            checks.appendChild(card);
        });
    }

    if (Array.isArray(data.sources) && data.sources.length) {
        const heading = document.createElement("h3");
        heading.textContent = "Manual search links";
        sourcesRoot.appendChild(heading);

        data.sources.forEach((source) => {
            addLink(
                sourcesRoot,
                source.name || "Search source",
                source.url,
                "manual-link"
            );
        });
    }
}

function render(saved) {
    $("loading").classList.add("hidden");
    $("error").classList.add("hidden");
    $("report").classList.remove("hidden");

    const analysis = saved.analysis || {};
    setText("originalText", saved.text);
    setText("predictionLabel", analysis.label || analysis.prediction);
    setText("confidenceValue",
        Number.isFinite(Number(analysis.confidence))
            ? `${(Number(analysis.confidence) * 100).toFixed(2)}%`
            : analysis.confidence
    );
    setText("labelDescription", analysis.label_description || analysis.description || analysis.message, "");

    const warning = $("warning");
    const warningText = analysis.input_warning || analysis.warning;
    warning.textContent = warningText || "";
    warning.classList.toggle("hidden", !warningText);

    setText("caution", analysis.caution || analysis.caution_message, "");
    setText("explainNote", analysis.explanation_note || analysis.explanation?.note, "");

    const explanation = analysis.explanation || {};
    renderProbabilities(analysis);
    renderClues("positiveClues",
        explanation.positive_clues || analysis.positive_clues);
    renderClues("negativeClues",
        explanation.negative_clues || analysis.negative_clues);
    renderEvidence(saved);
}

async function loadReport() {
    $("loading").classList.remove("hidden");
    $("report").classList.add("hidden");
    $("error").classList.add("hidden");

    try {
        const result = await chrome.storage.local.get("truthguardDetailedResult");
        if (!result.truthguardDetailedResult) {
            throw new Error("No saved analysis was found. Run Analyze Page or Check Selected Text first.");
        }
        render(result.truthguardDetailedResult);
    } catch (error) {
        $("loading").classList.add("hidden");
        $("error").textContent = error.message || "Could not load the saved analysis.";
        $("error").classList.remove("hidden");
    }
}

$("refreshBtn").addEventListener("click", loadReport);

chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && changes.truthguardDetailedResult) {
        loadReport();
    }
});

loadReport();
