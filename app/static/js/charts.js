/* Chart helpers for rendering values returned by the analysis API only. */
(() => {
    const charts = new Map();
    const palette = {
        green: "#07805f",
        coral: "#c65e4a",
        gold: "#b28b16",
        grid: "#e8eeea",
        text: "#6f7e77",
    };

    function draw(canvasId, config) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !window.Chart) return;
        charts.get(canvasId)?.destroy();
        charts.set(canvasId, new Chart(canvas, config));
    }

    function commonOptions(yTitle, percent = false) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { position: "bottom", labels: { color: palette.text, usePointStyle: true, boxWidth: 7, font: { family: "DM Mono", size: 9 } } },
                tooltip: { callbacks: percent ? { label: (context) => `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%` } : {} },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: palette.text, maxTicksLimit: 8, maxRotation: 0, font: { family: "DM Mono", size: 9 } } },
                y: { title: { display: true, text: yTitle, color: palette.text, font: { family: "DM Mono", size: 9 } }, grid: { color: palette.grid }, ticks: { color: palette.text, font: { family: "DM Mono", size: 9 }, callback: percent ? (value) => `${(value * 100).toFixed(0)}%` : undefined } },
            },
        };
    }

    window.StockCharts = {
        renderPrice(dates, close) {
            draw("price-chart", {
                type: "line",
                data: { labels: dates, datasets: [{ label: "Close", data: close, borderColor: palette.green, backgroundColor: "rgba(7,128,95,.10)", fill: true, pointRadius: 0, borderWidth: 2, tension: 0.12 }] },
                options: { ...commonOptions("Close"), plugins: { ...commonOptions("Close").plugins, legend: { display: false } } },
            });
        },
        renderDrawdowns(dates, actual, modelSeries) {
            const datasets = [{ label: "Actual test drawdown", data: actual, borderColor: palette.ink || "#17231f", backgroundColor: "#17231f", pointRadius: 2, borderWidth: 1.5, tension: 0 }];
            const colors = [palette.green, palette.coral];
            modelSeries.forEach((series, index) => datasets.push({ label: `${series.label} estimate`, data: series.values, borderColor: colors[index % colors.length], backgroundColor: colors[index % colors.length], pointRadius: 1.5, borderWidth: 1.5, tension: 0 }));
            draw("drawdown-chart", { type: "line", data: { labels: dates, datasets }, options: commonOptions("Drawdown", true) });
        },
        renderEvents(dates, actual, modelSeries) {
            const datasets = [{ label: "Actual event", data: actual, borderColor: "#17231f", backgroundColor: "#17231f", pointRadius: 2, borderWidth: 1.5, stepped: true }];
            const colors = [palette.green, palette.coral];
            modelSeries.forEach((series, index) => datasets.push({ label: `${series.label} predicted`, data: series.values, borderColor: colors[index % colors.length], backgroundColor: colors[index % colors.length], pointRadius: 2, borderWidth: 1.5, stepped: true }));
            draw("event-chart", {
                type: "line",
                data: { labels: dates, datasets },
                options: {
                    ...commonOptions("Event flag"),
                    scales: {
                        x: commonOptions("Event flag").scales.x,
                        y: { min: -0.1, max: 1.1, grid: { color: palette.grid }, ticks: { stepSize: 1, color: palette.text, font: { family: "DM Mono", size: 9 }, callback: (value) => value === 1 ? "Event" : value === 0 ? "No event" : "" } },
                    },
                },
            });
        },
        renderImportance(features, values) {
            draw("importance-chart", {
                type: "bar",
                data: { labels: features, datasets: [{ label: "Gain importance", data: values, backgroundColor: features.map((_, index) => index === 0 ? palette.green : "#8ab9a5"), borderRadius: 2, barThickness: 13 }] },
                options: { ...commonOptions("Relative importance"), indexAxis: "y", plugins: { ...commonOptions("Relative importance").plugins, legend: { display: false } }, scales: { x: { beginAtZero: true, grid: { color: palette.grid }, ticks: { color: palette.text, font: { family: "DM Mono", size: 9 } } }, y: { grid: { display: false }, ticks: { color: palette.text, font: { family: "DM Mono", size: 9 } } } } },
            });
        },
        renderMetricComparison(models) {
            const names = Object.keys(models);
            const metrics = ["mse", "mae", "rmse"];
            const colors = [palette.green, palette.coral];
            draw("metrics-chart", {
                type: "bar",
                data: {
                    labels: metrics.map((name) => name.toUpperCase()),
                    datasets: names.map((name, index) => ({
                        label: name.toUpperCase(),
                        data: metrics.map((metric) => models[name].metrics[metric]),
                        backgroundColor: colors[index % colors.length],
                        borderRadius: 2,
                    })),
                },
                options: {
                    ...commonOptions("Metric value"),
                    scales: {
                        x: { ...commonOptions("Metric value").scales.x, stacked: false },
                        y: { ...commonOptions("Metric value").scales.y, beginAtZero: true },
                    },
                },
            });
        },
    };
})();