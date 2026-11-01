/* Render persisted reports using textContent to keep report text inert. */
(() => {
    const body = document.getElementById("reports-body");
    if (!body) return;
    const count = document.getElementById("reports-count");
    const error = document.getElementById("reports-error");
    const compareButton = document.getElementById("compare-reports");
    const comparison = document.getElementById("comparison-result");
    const selectedIds = new Set();

    function updateCompareButton() {
        compareButton.disabled = selectedIds.size !== 2;
        compareButton.textContent = selectedIds.size === 2 ? "Compare selected" : `Select two reports (${selectedIds.size}/2)`;
    }

    function cell(row, value, className = "") {
        const td = document.createElement("td");
        td.className = className;
        td.textContent = value;
        row.append(td);
        return td;
    }

    async function loadReports() {
        error.textContent = "";
        try {
            const response = await fetch("/api/reports?limit=100");
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Could not load reports.");
            body.replaceChildren();
            const reports = payload.reports;
            selectedIds.clear();
            comparison.hidden = true;
            comparison.replaceChildren();
            updateCompareButton();
            count.textContent = `${reports.length} SAVED REPORT${reports.length === 1 ? "" : "S"}`;
            if (!reports.length) {
                const row = document.createElement("tr");
                cell(row, "No reports yet. Run an analysis from the dashboard.", "table-empty").colSpan = 7;
                body.append(row);
                return;
            }
            for (const report of reports) {
                const row = document.createElement("tr");
                const selectorCell = cell(row, "");
                const selector = document.createElement("input");
                selector.type = "checkbox";
                selector.className = "report-select";
                selector.setAttribute("aria-label", `Select report ${report.ticker} ${report.report_id}`);
                selector.addEventListener("change", () => {
                    if (selector.checked && selectedIds.size >= 2) {
                        selector.checked = false;
                        error.textContent = "Select no more than two reports.";
                        return;
                    }
                    if (selector.checked) selectedIds.add(report.report_id);
                    else selectedIds.delete(report.report_id);
                    error.textContent = "";
                    updateCompareButton();
                });
                selectorCell.append(selector);
                cell(row, report.ticker, "ticker-cell");
                cell(row, `${report.start_date} — ${report.end_date}`);
                cell(row, report.selected_model.toUpperCase());
                cell(row, `${(report.crisis_threshold * 100).toFixed(2)}% / ${report.prediction_horizon} sessions`);
                cell(row, new Date(report.created_at).toLocaleString());
                const action = cell(row, "");
                const link = document.createElement("a");
                link.className = "table-report-link";
                link.href = `/?report_id=${encodeURIComponent(report.report_id)}`;
                link.textContent = "Open";
                action.append(link);
                body.append(row);
            }
        } catch (cause) {
            count.textContent = "REPORTS UNAVAILABLE";
            error.textContent = cause.message || "Could not load reports.";
            body.replaceChildren();
            const row = document.createElement("tr");
            cell(row, "Reports could not be loaded.", "table-empty").colSpan = 7;
            body.append(row);
        }
    }

    async function compareSelected() {
        if (selectedIds.size !== 2) return;
        error.textContent = "";
        compareButton.disabled = true;
        compareButton.textContent = "Comparing…";
        try {
            const [reportId1, reportId2] = selectedIds;
            const response = await fetch("/api/reports/compare", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ report_id_1: reportId1, report_id_2: reportId2 }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload.error || "Could not compare reports.");
            comparison.replaceChildren();
            const heading = document.createElement("h2");
            heading.textContent = `Metric comparison · ${payload.comparison.tickers.join(" vs ")}`;
            comparison.append(heading);
            const table = document.createElement("table");
            table.className = "comparison-table";
            const header = document.createElement("tr");
            ["Model / metric", "Report 1", "Report 2", "Difference"].forEach((value) => {
                const th = document.createElement("th");
                th.textContent = value;
                header.append(th);
            });
            table.append(header);
            for (const [model, metrics] of Object.entries(payload.comparison.metrics)) {
                for (const [name, values] of Object.entries(metrics)) {
                    const row = document.createElement("tr");
                    [
                        `${model.toUpperCase()} ${name.toUpperCase()}`,
                        Number(values.first).toFixed(5),
                        Number(values.second).toFixed(5),
                        Number(values.difference_second_minus_first).toFixed(5),
                    ].forEach((value) => {
                        const td = document.createElement("td");
                        td.textContent = value;
                        row.append(td);
                    });
                    table.append(row);
                }
            }
            comparison.append(table);
            comparison.hidden = false;
        } catch (cause) {
            error.textContent = cause.message || "Could not compare reports.";
        } finally {
            updateCompareButton();
        }
    }

    document.getElementById("refresh-reports").addEventListener("click", loadReports);
    compareButton.addEventListener("click", compareSelected);
    loadReports();
})();