from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

from graph.graph_store import get_course_graph_path
from graph.graph_store import load_module_graph
from graph.module_graph_builder import build_course_module_graph_for_id


BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    """Entry point for rebuilding, summarizing, and optionally serving the graph."""
    parser = argparse.ArgumentParser(description="Debug and render the module knowledge graph.")
    parser.add_argument("course_id", nargs="?", default="math")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the graph before rendering.")
    parser.add_argument("--serve", action="store_true", help="Serve the rendered HTML on localhost.")
    parser.add_argument(
        "--module",
        default=None,
        help="Focus the viewer on one module id, for example math:2.4.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Local port for the debug viewer.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the localhost viewer in the default browser after starting the server.",
    )
    args = parser.parse_args()

    graph = (
        build_course_module_graph_for_id(args.course_id)
        if args.rebuild or not get_course_graph_path(args.course_id).exists()
        else load_module_graph(args.course_id)
    )

    print("=" * 80)
    print("MODULE GRAPH DEBUG")
    print("=" * 80)
    print(f"Course: {graph['course_title']} ({graph['course_id']})")
    print(f"Graph version: {graph['graph_version']}")
    print(f"Nodes: {graph['node_count']}")
    print(f"Edges: {graph['edge_count']}")

    # Print a compact summary first so the graph can be sanity-checked from the
    # terminal without requiring the HTML viewer.
    _print_relation_summary(graph)

    focused_module_id = args.module or graph["nodes"][0]["id"]
    print(f"Focused module: {focused_module_id}")

    html_path = _write_graph_view(graph, focused_module_id)
    print(f"\nHTML viewer written to: {html_path}")

    if not args.serve:
        print("Run with --serve to render this graph on localhost.")
        return 0

    return _serve_graph_view(html_path, port=args.port, open_browser=args.open_browser)


def _print_relation_summary(graph: dict) -> None:
    """Keep terminal output short but enough for a quick sanity check."""
    relation_counts: dict[str, int] = {}
    for edge in graph["edges"]:
        relation = edge["relation"]
        relation_counts[relation] = relation_counts.get(relation, 0) + 1

    print("\nRelation counts:")
    for relation, count in sorted(relation_counts.items()):
        print(f"- {relation}: {count}")

    print("\nSample edges:")
    for edge in graph["edges"][:10]:
        print(
            f"- {edge['source']} -> {edge['target']} "
            f"[{edge['relation']}, weight={edge['weight']}]"
        )


def _write_graph_view(graph: dict, focused_module_id: str) -> Path:
    """Write the standalone HTML viewer used for local graph inspection."""
    graph_dir = get_course_graph_path(graph["course_id"]).parent
    html_path = graph_dir / f"{graph['course_id']}_module_graph_view.html"
    html_path.write_text(_build_graph_html(graph, focused_module_id), encoding="utf-8")
    return html_path


def _build_graph_html(graph: dict, focused_module_id: str) -> str:
    graph_json = json.dumps(graph, ensure_ascii=False)
    module_options = "\n".join(
        f'<option value="{node["id"]}"{" selected" if node["id"] == focused_module_id else ""}>'
        f'{node["id"]} - {node["title"]}</option>'
        for node in graph["nodes"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Module Graph Debug</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f1ea;
      color: #1f2937;
    }}
    .app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid #d6d3d1;
      padding: 16px;
      background: #fffaf4;
      overflow: auto;
    }}
    .canvas-wrap {{
      position: relative;
      overflow: auto;
      background: linear-gradient(180deg, #fffdf8 0%, #f7efe4 100%);
    }}
    .section-title {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #6b7280;
      margin: 16px 0 8px 0;
    }}
    .stat {{
      margin: 4px 0;
      font-size: 14px;
    }}
    .filter-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      font-size: 14px;
    }}
    .details {{
      margin-top: 12px;
      padding: 12px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.5;
    }}
    select {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid #d6d3d1;
      background: white;
      font-size: 14px;
    }}
    svg {{
      width: 1600px;
      height: 1000px;
      display: block;
    }}
    .node {{
      cursor: pointer;
    }}
    .node rect {{
      fill: #fff;
      stroke: #c08457;
      stroke-width: 1.5;
      rx: 14;
      ry: 14;
    }}
    .node.active rect {{
      fill: #fff7ed;
      stroke: #ea580c;
      stroke-width: 2.5;
    }}
    .node text {{
      font-size: 12px;
      fill: #1f2937;
      pointer-events: none;
    }}
    .edge {{
      fill: none;
      stroke-width: 2;
      opacity: 0.55;
    }}
    .edge.prerequisite_of {{
      stroke: #2563eb;
    }}
    .edge.similar_to {{
      stroke: #7c3aed;
    }}
    .helper-text {{
      color: #6b7280;
      font-size: 13px;
      line-height: 1.5;
      margin-top: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      margin: 6px 0;
    }}
    .legend-line {{
      width: 28px;
      height: 0;
      border-top: 3px solid;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(17, 24, 39, 0.92);
      color: white;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 12px;
      line-height: 1.5;
      max-width: 320px;
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
      display: none;
      z-index: 20;
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="section-title">Graph</div>
      <div class="stat"><strong>Course:</strong> {graph["course_title"]}</div>
      <div class="stat"><strong>Version:</strong> {graph["graph_version"]}</div>
      <div class="stat"><strong>Nodes:</strong> {graph["node_count"]}</div>
      <div class="stat"><strong>Edges:</strong> {graph["edge_count"]}</div>

      <div class="section-title">Filters</div>
      <label class="filter-row">
        <input type="checkbox" id="showPrereq" checked />
        prerequisite_of
      </label>
      <label class="filter-row">
        <input type="checkbox" id="showSimilar" checked />
        similar_to
      </label>

      <div class="section-title">Focus Module</div>
      <select id="moduleSelect">
        {module_options}
      </select>
      <div class="helper-text">
        The viewer renders the selected module and its one-hop neighbors only.
      </div>

      <div class="section-title">Legend</div>
      <div class="legend-item">
        <span class="legend-line" style="border-color:#2563eb;"></span>
        prerequisite_of
      </div>
      <div class="legend-item">
        <span class="legend-line" style="border-color:#7c3aed;"></span>
        similar_to
      </div>

      <div class="section-title">Node Details</div>
      <div id="details" class="details">Click a node to inspect module metadata and connected edges.</div>
    </aside>

    <main class="canvas-wrap">
      <svg id="graph" viewBox="0 0 1600 1000"></svg>
      <div id="tooltip" class="tooltip"></div>
    </main>
  </div>

  <script>
    const graph = {graph_json};
    const svg = document.getElementById("graph");
    const details = document.getElementById("details");
    const moduleSelect = document.getElementById("moduleSelect");
    const tooltip = document.getElementById("tooltip");
    const relationFilters = {{
      prerequisite_of: document.getElementById("showPrereq"),
      similar_to: document.getElementById("showSimilar"),
    }};
    const nodes = graph.nodes.map(node => ({{ ...node }}));
    const nodeMap = new Map(nodes.map(node => [node.id, node]));
    let activeNodeId = moduleSelect.value;

    function ensureDefs() {{
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.innerHTML = `
        <marker id="arrow-prerequisite" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#2563eb"></path>
        </marker>
        <marker id="arrow-similar" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#7c3aed"></path>
        </marker>
      `;
      svg.appendChild(defs);
    }}

    function edgeVisible(edge) {{
      const checkbox = relationFilters[edge.relation];
      return checkbox ? checkbox.checked : true;
    }}

    function getFocusedSubgraph(centerId) {{
      // The viewer stays intentionally local: only the focused module plus
      // one-hop neighbors are shown, which keeps the graph readable.
      const visibleEdges = graph.edges.filter(edgeVisible).filter(edge => edge.source === centerId || edge.target === centerId);
      const visibleIds = new Set([centerId]);
      visibleEdges.forEach(edge => {{
        visibleIds.add(edge.source);
        visibleIds.add(edge.target);
      }});

      return {{
        nodes: nodes.filter(node => visibleIds.has(node.id)),
        edges: visibleEdges,
      }};
    }}

    function render() {{
      svg.innerHTML = "";
      ensureDefs();
      const centerId = moduleSelect.value;
      const subgraph = getFocusedSubgraph(centerId);
      const centerNode = nodeMap.get(centerId);

      const positionedNodes = [];
      if (centerNode) {{
        positionedNodes.push({{ ...centerNode, x: 680, y: 430, lane: "center" }});
      }}

      // Directed edges are laid out left-to-right to make learning order easy to read.
      const incoming = subgraph.edges.filter(edge => edge.target === centerId);
      const outgoing = subgraph.edges.filter(edge => edge.source === centerId);

      // Put likely prerequisites on the left so "incoming knowledge" reads naturally.
      incoming.forEach((edge, index) => {{
        const node = nodeMap.get(edge.source);
        if (!node || node.id === centerId) return;
        positionedNodes.push({{ ...node, x: 180, y: 180 + index * 110, lane: "incoming" }});
      }});

      // Put direct follow-up modules on the right.
      outgoing.forEach((edge, index) => {{
        const node = nodeMap.get(edge.target);
        if (!node || node.id === centerId) return;
        positionedNodes.push({{ ...node, x: 1180, y: 180 + index * 110, lane: "outgoing" }});
      }});

      // Undirected similarity edges do not imply before/after, so park them above the center.
      const positionedIds = new Set(positionedNodes.map(node => node.id));
      const similarNeighbors = subgraph.edges
        .filter(edge => edge.relation === "similar_to")
        .map(edge => edge.source === centerId ? edge.target : edge.source)
        .filter(id => id !== centerId && !positionedIds.has(id));
      similarNeighbors.forEach((id, index) => {{
        const node = nodeMap.get(id);
        if (!node) return;
        positionedNodes.push({{ ...node, x: 680, y: 120 + index * 110, lane: "similar" }});
        positionedIds.add(id);
      }});

      const positionedMap = new Map(positionedNodes.map(node => [node.id, node]));

      subgraph.edges.forEach(edge => {{
        const source = positionedMap.get(edge.source);
        const target = positionedMap.get(edge.target);
        if (!source || !target) return;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const startX = source.x + 120;
        const startY = source.y + 24;
        const endX = target.x;
        const endY = target.y + 24;
        const controlX = (startX + endX) / 2;
        const d = `M ${{startX}} ${{startY}} C ${{controlX}} ${{startY}}, ${{controlX}} ${{endY}}, ${{endX}} ${{endY}}`;
        path.setAttribute("d", d);
        path.setAttribute("class", `edge ${{edge.relation}}`);
        path.setAttribute("data-source", edge.source);
        path.setAttribute("data-target", edge.target);
        path.setAttribute("data-relation", edge.relation);
        if (edge.directed) {{
          path.setAttribute(
            "marker-end",
            edge.relation === "prerequisite_of"
              ? "url(#arrow-prerequisite)"
              : "url(#arrow-similar)"
          );
        }}
        path.addEventListener("mousemove", event => showEdgeTooltip(event, edge));
        path.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(path);
      }});

      positionedNodes.forEach(node => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", `node${{activeNodeId === node.id ? " active" : ""}}`);
        group.setAttribute("transform", `translate(${{node.x}}, ${{node.y}})`);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("width", "240");
        rect.setAttribute("height", "48");
        group.appendChild(rect);

        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        title.setAttribute("x", "12");
        title.setAttribute("y", "20");
        title.textContent = `${{node.toc_number || node.id}}  ${{node.title}}`;
        group.appendChild(title);

        const page = document.createElementNS("http://www.w3.org/2000/svg", "text");
        page.setAttribute("x", "12");
        page.setAttribute("y", "36");
        page.textContent = `page ${{node.start_page || "?"}}`;
        group.appendChild(page);

        group.addEventListener("click", () => showNodeDetails(node.id));
        svg.appendChild(group);
      }});
    }}

    function showEdgeTooltip(event, edge) {{
      tooltip.style.display = "block";
      tooltip.style.left = `${{event.clientX + 14}}px`;
      tooltip.style.top = `${{event.clientY + 14}}px`;
      tooltip.textContent =
`Relation: ${{edge.relation}}
Directed: ${{edge.directed ? "yes" : "no"}}
Weight: ${{edge.weight}}
Source: ${{edge.source}}
Target: ${{edge.target}}
Reason: ${{edge.reason}}`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function showNodeDetails(nodeId) {{
      activeNodeId = nodeId;
      moduleSelect.value = nodeId;
      const node = nodeMap.get(nodeId);
      // Show all graph edges for the selected node, even if a filter is hidden,
      // so the details panel stays a faithful record of the stored graph.
      const outgoing = graph.edges.filter(edge => edge.source === nodeId);
      const incoming = graph.edges.filter(edge => edge.target === nodeId);

      details.textContent =
`Module
${{node.id}} - ${{node.title}}

Chapter
${{node.chapter_title || "Unknown"}}

Primary skill
${{node.primary_skill}}

Start page
${{node.start_page || "Unknown"}}

Outgoing edges
${{outgoing.map(edge => `- ${{edge.relation}} -> ${{edge.target}} (weight=${{edge.weight}}) | ${{edge.reason}}`).join("\\n") || "None"}}

Incoming edges
${{incoming.map(edge => `- ${{edge.relation}} <- ${{edge.source}} (weight=${{edge.weight}}) | ${{edge.reason}}`).join("\\n") || "None"}}`;
      render();
    }}

    Object.values(relationFilters).forEach(checkbox => {{
      checkbox.addEventListener("change", render);
    }});
    moduleSelect.addEventListener("change", () => showNodeDetails(moduleSelect.value));

    showNodeDetails(moduleSelect.value);
  </script>
</body>
</html>
"""


def _serve_graph_view(html_path: Path, port: int, open_browser: bool) -> int:
    """Serve the generated HTML from the project root so linked assets resolve cleanly."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    with TCPServer(("127.0.0.1", port), Handler) as httpd:
        relative_path = html_path.relative_to(BASE_DIR).as_posix()
        url = f"http://127.0.0.1:{port}/{relative_path}"
        print(f"Serving graph viewer at: {url}")

        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nGraph viewer stopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
