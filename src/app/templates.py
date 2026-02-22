"""HTML templates for web endpoints."""


def get_graph_viewer_html(mermaid_diagram: str) -> str:
    """Generate HTML page for the interactive graph viewer."""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Agent - Graph Visualization</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a3c5e;
            margin-top: 0;
        }}
        .description {{
            color: #666;
            margin-bottom: 30px;
            line-height: 1.6;
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 4px;
        }}
        .node-legend {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .node-legend h3 {{
            margin-top: 0;
            color: #1a3c5e;
        }}
        .node-legend ul {{
            list-style: none;
            padding: 0;
        }}
        .node-legend li {{
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .node-legend li:last-child {{
            border-bottom: none;
        }}
        code {{
            background: #e8eaf6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🗺️ Travel Recommendation Agent - Pipeline Graph</h1>
        <p class="description">
            This diagram shows the LangGraph state machine that powers the travel recommendation system.
            Each node represents a processing step, and edges show the flow of data through the pipeline.
        </p>
        
        <div class="mermaid">
{mermaid_diagram}
        </div>
        
        <div class="node-legend">
            <h3>Pipeline Nodes</h3>
            <ul>
                <li><code>validate</code> — Validates user input (IATA codes, dates, budget)</li>
                <li><code>suggest</code> — LLM suggests 5-8 destinations based on preferences</li>
                <li><code>search_flights</code> — Queries Amadeus API for available flights</li>
                <li><code>increment_retry</code> — Increments retry counter (if no flights found)</li>
                <li><code>enrich</code> — Adds weather data and activity recommendations</li>
                <li><code>generate_report</code> — Creates the comparative PDF report</li>
            </ul>
            <p style="margin-top: 15px; color: #666;">
                <strong>Conditional edges:</strong> The pipeline can retry destination suggestions once 
                if no affordable flights are found (search_flights → increment_retry → suggest).
            </p>
        </div>
    </div>
</body>
</html>"""
