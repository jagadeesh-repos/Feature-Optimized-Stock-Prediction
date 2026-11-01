/* Submit real analysis requests and render only returned report data. */
(() => {
    const form = document.getElementById("analysis-form");
    if (!form) return;

    const button = document.getElementById("analyze-button");
    const buttonLabel = button.querySelector(".button-label");
    const message = document.getElementById("analysis-message");
    const percent = (value) => `${(Number(value) * 100).toFixed(2)}%`;
    const metric = (value) => Number(value).toFixed(5);

    function addText(parent, tag, className, value) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        element.textContent = value;
        parent.append(element);
        return element;
    }

    function renderSummary(report) {
        const target = document.getElementById("model-summary");
        target.replaceChildren();
        for (const [name, result] of Object.entries(report.models || {})) {
            const item = document.createElement("div");
            item.className = "summary-item";
            addText(item, "span", "summary-label", `${name.toUpperCase()} ESTIMATED DRAWDOWN`);
            const value = addText(item, "strong", "summary-value", percent(result.latest_predicted_drawdown));
            value.classList.add(result.latest_predicted_crisis ? "is-alert" : "is-clear");
            addText(item, "span", "summary-meta", result.latest_predicted_crisis ? "At or above selected threshold" : "Below selected threshold");
            target.append(item);
        }
        const latest = document.createElement("div");
        latest.className = "summary-item";
        const prices = report.predictions.market_history.close;
        addText(latest, "span", "summary-label", "LATEST AVAILABLE CLOSE");
        addText(latest, "strong", "summary-value", Number(prices[prices.length - 1]).toLocaleString(undefined, { maximumFractionDigits: 2 }));
        addText(latest, "span", "summary-meta", `${report.ticker} · forecast-only row`);
        target.append(latest);
    }

    function renderMetrics(models) {
        const target = document.getElementById("metrics-table");
        target.replaceChildren();
        const headings = ["Model", "MSE", "MAE", "RMSE"];
        const header = document.createElement("div");
        header.className = "metrics-row header";
        headings.forEach((heading) => addText(header, "span", "", heading));
        target.append(header);
        for (const [name, result] of Object.entries(models)) {
            const row = document.createElement("div");
            row.className = "metrics-row";
            addText(row, "span", "model-cell", name.toUpperCase());
            ["mse", "mae", "rmse"].forEach((key) => addText(row, "span", "", metric(result.metrics[key])));
            target.append(row);
        }
    }

    function render(report) {
        document.getElementById("empty-state").hidden = true;
        document.getElementById("results").hidden = false;
        document.getElementById("result-ticker").textContent = report.ticker;
        document.getElementById("result-period").textContent = `${report.start_date} — ${report.end_date} · ${report.prediction_horizon} sessions · ${percent(report.crisis_threshold)} threshold`;
        document.getElementById("report-link").href = `/?report_id=${encodeURIComponent(report.report_id)}`;
        renderSummary(report);
        renderMetrics(report.models);
        window.StockCharts?.renderMetricComparison(report.models);
        const selectedFeatures = document.getElementById("selected-feature-list");
        selectedFeatures.replaceChildren();
        report.selected_features.forEach((feature) => addText(selectedFeatures, "li", "", feature));

        const predictions = report.predictions;
        window.StockCharts?.renderPrice(
            predictions.market_history.dates.map((date) => new Date(date).toLocaleDateString()),
            predictions.market_history.close,
        );
        const models = Object.entries(report.models).map(([name, result]) => ({
            label: name.toUpperCase(),
            values: result.test_predictions,
        }));
        window.StockCharts?.renderDrawdowns(
            predictions.test_dates.map((date) => new Date(date).toLocaleDateString()),
            predictions.actual_test_drawdowns,
            models,
        );
        window.StockCharts?.renderEvents(
            predictions.test_dates.map((date) => new Date(date).toLocaleDateString()),
            predictions.actual_test_crisis_events,
            Object.entries(report.models).map(([name, result]) => ({
                label: name.toUpperCase(),
                values: result.test_predicted_crisis_events,
            })),
        );

        const importance = report.models.xgboost?.feature_importance;
        if (importance && Object.keys(importance).length) {
            const entries = Object.entries(importance).sort((left, right) => right[1] - left[1]);
            document.getElementById("importance-note").textContent = "XGBOOST GAIN";
            document.getElementById("importance-chart").hidden = false;
            window.StockCharts?.renderImportance(entries.map(([name]) => name), entries.map(([, value]) => value));
        } else {
            document.getElementById("importance-note").textContent = "Not provided by this model";
            document.getElementById("importance-chart").hidden = true;
        }
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        message.textContent = "";
        if (!form.reportValidity()) return;
        const thresholdPercent = Number(form.elements.crisis_threshold_percent.value);
        if (!(thresholdPercent > 0 && thresholdPercent < 100)) {
            message.textContent = "Threshold must be between 0 and 100 percent.";
            return;
        }

        button.disabled = true;
        buttonLabel.textContent = "Fetching data and training…";
        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ticker: form.elements.ticker.value.trim(),
                    start_date: form.elements.start_date.value,
                    end_date: form.elements.end_date.value,
                    crisis_threshold: thresholdPercent / 100,
                    prediction_horizon: Number(form.elements.prediction_horizon.value),
                    selected_model: form.elements.selected_model.value,
                }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Analysis request failed.");
            render(payload.report);
            message.textContent = `Report saved: ${payload.report.report_id}`;
            message.classList.add("success-message");
        } catch (error) {
            message.classList.remove("success-message");
            message.textContent = error.message || "Could not complete the analysis.";
        } finally {
            button.disabled = false;
            buttonLabel.textContent = "Run analysis";
        }
    });

    const savedReportId = new URLSearchParams(window.location.search).get("report_id");
    if (savedReportId) {
        fetch(`/api/reports/${encodeURIComponent(savedReportId)}`)
            .then(async (response) => {
                const payload = await response.json();
                if (!response.ok) throw new Error(payload.error || "Could not load report.");
                render(payload.report);
                message.textContent = `Loaded saved report: ${payload.report.report_id}`;
            })
            .catch((cause) => {
                message.textContent = cause.message || "Could not load saved report.";
            });
    }
})();