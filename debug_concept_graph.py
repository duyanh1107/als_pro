from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

from graph.concept_graph_builder import build_catalog_concept_graph
from graph.concept_graph_builder import build_course_concept_graph_for_id
from graph.graph_store import get_concept_graph_path
from graph.graph_store import load_concept_graph


BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    """Entry point for rebuilding, summarizing, and optionally serving the concept graph."""
    parser = argparse.ArgumentParser(description="Debug and render the concept knowledge graph.")
    parser.add_argument("scope_id", nargs="?", default="math", help="Course id like math or the special scope 'catalog'.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the graph before rendering.")
    parser.add_argument("--list-concepts", action="store_true", help="Print every concept node in the terminal.")
    parser.add_argument("--serve", action="store_true", help="Serve the rendered HTML on localhost.")
    parser.add_argument(
        "--focus",
        default=None,
        help="Focus the viewer on one node id, for example math:2.2 or concept:matrix.",
    )
    parser.add_argument("--port", type=int, default=8766, help="Local port for the debug viewer.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the localhost viewer in the default browser after starting the server.",
    )
    args = parser.parse_args()

    graph = (
        _rebuild_graph(args.scope_id)
        if args.rebuild or not get_concept_graph_path(args.scope_id).exists()
        else load_concept_graph(args.scope_id)
    )

    print("=" * 80)
    print("CONCEPT GRAPH DEBUG")
    print("=" * 80)
    print(f"Scope: {graph['scope']} ({graph['scope_id']})")
    print(f"Version: {graph['graph_version']}")
    print(f"Nodes: {graph['node_count']}")
    print(f"Edges: {graph['edge_count']}")
    print(f"Modules: {graph['module_count']}")
    print(f"Concepts: {graph['concept_count']}")

    _print_relation_summary(graph)
    if args.list_concepts:
        _print_concept_list(graph)

    focused_node_id = args.focus or _default_focus_node(graph)
    print(f"Focused node: {focused_node_id}")

    html_path = _write_graph_view(graph, focused_node_id)
    print(f"\nHTML viewer written to: {html_path}")

    if not args.serve:
        print("Run with --serve to render this graph on localhost.")
        return 0

    return _serve_graph_view(html_path, port=args.port, open_browser=args.open_browser)


def _rebuild_graph(scope_id: str) -> dict:
    if scope_id == "catalog":
        return build_catalog_concept_graph()
    return build_course_concept_graph_for_id(scope_id)


def _default_focus_node(graph: dict) -> str:
    # Prefer a module as the first focus because modules are easier for humans to recognize.
    for node in graph["nodes"]:
        if node["type"] == "module":
            return node["id"]
    return graph["nodes"][0]["id"]


def _print_relation_summary(graph: dict) -> None:
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


def _print_concept_list(graph: dict) -> None:
    print("\nConcept nodes:")
    concepts = sorted(
        (node for node in graph["nodes"] if node["type"] == "concept"),
        key=lambda node: node["label"],
    )
    for concept in concepts:
        print(f"- {concept['label']} ({concept['id']})")


def _write_graph_view(graph: dict, focused_node_id: str) -> Path:
    graph_dir = get_concept_graph_path(graph["scope_id"]).parent
    html_path = graph_dir / f"{graph['scope_id']}_concept_graph_view.html"
    html_path.write_text(_build_graph_html(graph, focused_node_id), encoding="utf-8")
    return html_path


def _build_graph_html(graph: dict, focused_node_id: str) -> str:
    graph_json = json.dumps(graph, ensure_ascii=False)
    concept_items = "\n".join(
        f'<button class="concept-item" data-node-id="{node["id"]}">{node["label"]} '
        f'<span class="concept-id">({node["id"]})</span></button>'
        for node in sorted(
            (node for node in graph["nodes"] if node["type"] == "concept"),
            key=lambda node: node["label"],
        )
    )
    node_options = "\n".join(
        f'<option value="{node["id"]}"{" selected" if node["id"] == focused_node_id else ""}>'
        f'{node["id"]} - {node.get("title") or node.get("label")}</option>'
        for node in graph["nodes"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Concept Graph Debug</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f1ea;
      color: #1f2937;
    }}
    .app {{
      display: grid;
      grid-template-columns: 340px 1fr;
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
    .canvas-controls {{
      position: sticky;
      top: 12px;
      right: 12px;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      padding: 12px;
      z-index: 8;
      pointer-events: none;
    }}
    .canvas-button {{
      pointer-events: auto;
      border: 1px solid #d6d3d1;
      background: rgba(255, 250, 244, 0.95);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
      color: #1f2937;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }}
    .canvas-button:hover {{
      background: #ffffff;
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
    .concept-list {{
      margin-top: 12px;
      max-height: 280px;
      overflow: auto;
      padding: 8px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
    }}
    .concept-item {{
      display: block;
      width: 100%;
      text-align: left;
      border: 0;
      background: transparent;
      padding: 8px 10px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      color: #1f2937;
    }}
    .concept-item:hover {{
      background: #f3f4f6;
    }}
    .concept-id {{
      color: #6b7280;
      font-size: 12px;
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
      width: 2400px;
      height: 1800px;
      display: block;
    }}
    .node {{
      cursor: pointer;
    }}
    .node rect {{
      stroke-width: 1.5;
      rx: 14;
      ry: 14;
    }}
    .node.module rect {{
      fill: #ffffff;
      stroke: #2563eb;
    }}
    .node.concept rect {{
      fill: #faf5ff;
      stroke: #7c3aed;
    }}
    .node.active rect {{
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
      opacity: 0.6;
    }}
    .edge.module_teaches_concept {{
      stroke: #059669;
    }}
    .edge.concept_prerequisite_of {{
      stroke: #ea580c;
    }}
    .edge.concept_related_to {{
      stroke: #0f766e;
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
      width: 32px;
      height: 0;
      border-top: 3px solid;
    }}
    .legend-line.dashed {{
      border-top-style: dashed;
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
      max-width: 340px;
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
      <div class="stat"><strong>Scope:</strong> {graph["scope_id"]}</div>
      <div class="stat"><strong>Version:</strong> {graph["graph_version"]}</div>
      <div class="stat"><strong>Nodes:</strong> {graph["node_count"]}</div>
      <div class="stat"><strong>Edges:</strong> {graph["edge_count"]}</div>
      <div class="stat"><strong>Modules:</strong> {graph["module_count"]}</div>
      <div class="stat"><strong>Concepts:</strong> {graph["concept_count"]}</div>

      <div class="section-title">Filters</div>
      <label class="filter-row">
        <input type="checkbox" id="showTeach" checked />
        module_teaches_concept
      </label>
      <label class="filter-row">
        <input type="checkbox" id="showConceptPrereq" checked />
        concept_prerequisite_of
      </label>
      <label class="filter-row">
        <input type="checkbox" id="showConceptRelated" checked />
        concept_related_to
      </label>
      <label class="filter-row">
        <input type="checkbox" id="showFullConceptMap" />
        full concept map
      </label>
      <label class="filter-row">
        <input type="checkbox" id="showSphereMode" />
        3d concept sphere
      </label>

      <div class="section-title">Focus Node</div>
      <select id="nodeSelect">
        {node_options}
      </select>
      <div class="helper-text">
        Use the default focused view for one-hop inspection, or enable full concept map
        to render all concept nodes in a compact radial layout. Turn on 3d concept sphere
        if you want the full map to orbit like a globe.
      </div>

      <div class="section-title">Legend</div>
      <div class="legend-item">
        <span class="legend-line" style="border-color:#059669;"></span>
        module_teaches_concept
      </div>
      <div class="legend-item">
        <span class="legend-line" style="border-color:#ea580c;"></span>
        concept_prerequisite_of
      </div>
      <div class="legend-item">
        <span class="legend-line dashed" style="border-color:#0f766e;"></span>
        concept_related_to
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:14px;height:14px;border:2px solid #2563eb;border-radius:4px;background:#fff;"></span>
        module node
      </div>
      <div class="legend-item">
        <span style="display:inline-block;width:14px;height:14px;border:2px solid #7c3aed;border-radius:4px;background:#faf5ff;"></span>
        concept node
      </div>

      <div class="section-title">Node Details</div>
      <div id="details" class="details">Click a node to inspect metadata and connected edges.</div>

      <div class="section-title">All Concept Nodes</div>
      <div id="conceptList" class="concept-list">
        {concept_items}
      </div>
    </aside>

    <main class="canvas-wrap">
      <div class="canvas-controls">
        <button id="fitViewButton" class="canvas-button" type="button">fit view</button>
        <button id="toggleRotateButton" class="canvas-button" type="button">pause rotation</button>
      </div>
      <svg id="graph" viewBox="0 0 1800 1100"></svg>
      <div id="tooltip" class="tooltip"></div>
    </main>
  </div>

  <script>
    const graph = {graph_json};
    const svg = document.getElementById("graph");
    const details = document.getElementById("details");
    const nodeSelect = document.getElementById("nodeSelect");
    const conceptList = document.getElementById("conceptList");
    const tooltip = document.getElementById("tooltip");
    const showFullConceptMap = document.getElementById("showFullConceptMap");
    const showSphereMode = document.getElementById("showSphereMode");
    const fitViewButton = document.getElementById("fitViewButton");
    const toggleRotateButton = document.getElementById("toggleRotateButton");
    const relationFilters = {{
      module_teaches_concept: document.getElementById("showTeach"),
      concept_prerequisite_of: document.getElementById("showConceptPrereq"),
      concept_related_to: document.getElementById("showConceptRelated"),
    }};
    const nodes = graph.nodes.map(node => ({{ ...node }}));
    const nodeMap = new Map(nodes.map(node => [node.id, node]));
    let activeNodeId = nodeSelect.value;
    const defaultViewBox = "0 0 1800 1100";
    let sphereAnimationFrame = null;
    let sphereRotation = 0;
    let sphereTilt = 0.45;
    let sphereRotationPaused = false;

    function ensureDefs() {{
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.innerHTML = `
        <marker id="arrow-teaches" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#059669"></path>
        </marker>
        <marker id="arrow-concept-prereq" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#ea580c"></path>
        </marker>
        <marker id="arrow-concept-related" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#0f766e"></path>
        </marker>
      `;
      svg.appendChild(defs);
    }}

    function edgeVisible(edge) {{
      const checkbox = relationFilters[edge.relation];
      return checkbox ? checkbox.checked : true;
    }}

    function getFocusedSubgraph(centerId) {{
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

    function getFullConceptSubgraph() {{
      // Full-map mode is concept-only on purpose. It always shows every concept
      // node, even if that concept has no prerequisite edge yet.
      const visibleEdges = graph.edges
        .filter(edgeVisible)
        .filter(edge => edge.relation === "concept_prerequisite_of" || edge.relation === "concept_related_to");
      const conceptNodes = nodes.filter(node => node.type === "concept");
      return {{
        nodes: conceptNodes,
        edges: visibleEdges,
      }};
    }}

    function render() {{
      stopSphereAnimation();
      svg.innerHTML = "";
      ensureDefs();
      svg.setAttribute("viewBox", defaultViewBox);
      const centerId = nodeSelect.value;
      if (showFullConceptMap.checked) {{
        if (showSphereMode.checked) {{
          renderConceptSphere();
        }} else {{
          renderFullConceptMap();
        }}
        return;
      }}

      const subgraph = getFocusedSubgraph(centerId);
      const centerNode = nodeMap.get(centerId);
      const positionedNodes = [];
      if (centerNode) {{
        positionedNodes.push({{ ...centerNode, x: 780, y: 460, lane: "center" }});
      }}

      const incoming = subgraph.edges.filter(edge => edge.target === centerId);
      const outgoing = subgraph.edges.filter(edge => edge.source === centerId);
      const relatedConcepts = subgraph.edges
        .filter(edge => edge.relation === "concept_related_to")
        .map(edge => edge.source === centerId ? edge.target : edge.source)
        .filter(id => id !== centerId);

      // Keep left = incoming, right = outgoing for any directed edge. This makes
      // both module-to-concept and concept prerequisite links easy to read.
      incoming.forEach((edge, index) => {{
        const node = nodeMap.get(edge.source);
        if (!node || node.id === centerId) return;
        positionedNodes.push({{ ...node, x: 220, y: 180 + index * 120, lane: "incoming" }});
      }});

      outgoing.forEach((edge, index) => {{
        const node = nodeMap.get(edge.target);
        if (!node || node.id === centerId) return;
        positionedNodes.push({{ ...node, x: 1300, y: 180 + index * 120, lane: "outgoing" }});
      }});

      const positionedIds = new Set(positionedNodes.map(node => node.id));
      relatedConcepts.forEach((id, index) => {{
        if (positionedIds.has(id)) return;
        const node = nodeMap.get(id);
        if (!node) return;
        positionedNodes.push({{ ...node, x: 780, y: 120 + index * 90, lane: "related" }});
        positionedIds.add(id);
      }});

      const positionedMap = new Map(positionedNodes.map(node => [node.id, node]));

      subgraph.edges.forEach(edge => {{
        const source = positionedMap.get(edge.source);
        const target = positionedMap.get(edge.target);
        if (!source || !target) return;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const startX = source.x + 130;
        const startY = source.y + 26;
        const endX = target.x;
        const endY = target.y + 26;
        const controlX = (startX + endX) / 2;
        const d = `M ${{startX}} ${{startY}} C ${{controlX}} ${{startY}}, ${{controlX}} ${{endY}}, ${{endX}} ${{endY}}`;
        path.setAttribute("d", d);
        path.setAttribute("class", `edge ${{edge.relation}}`);
        if (edge.directed) {{
          path.setAttribute(
            "marker-end",
            edge.relation === "module_teaches_concept"
              ? "url(#arrow-teaches)"
              : "url(#arrow-concept-prereq)"
          );
        }} else if (edge.relation === "concept_related_to") {{
          path.setAttribute("stroke-dasharray", "6 4");
        }}
        path.addEventListener("mousemove", event => showEdgeTooltip(event, edge));
        path.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(path);
      }});

      positionedNodes.forEach(node => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", `node ${{node.type}}${{activeNodeId === node.id ? " active" : ""}}`);
        group.setAttribute("transform", `translate(${{node.x}}, ${{node.y}})`);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("width", "260");
        rect.setAttribute("height", "52");
        group.appendChild(rect);

        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        title.setAttribute("x", "12");
        title.setAttribute("y", "21");
        title.textContent = node.type === "module"
          ? `${{node.toc_number || node.id}}  ${{node.title}}`
          : `${{node.id}}`;
        group.appendChild(title);

        const subtitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
        subtitle.setAttribute("x", "12");
        subtitle.setAttribute("y", "39");
        subtitle.textContent = node.type === "module"
          ? `chapter: ${{node.chapter_title || "unknown"}}`
          : `${{node.label}}`;
        group.appendChild(subtitle);

        group.addEventListener("click", () => showNodeDetails(node.id));
        svg.appendChild(group);
      }});
    }}

    function renderFullConceptMap() {{
      const subgraph = getFullConceptSubgraph();
      const positionedNodes = [];
      const centerX = 1180;
      const centerY = 860;
      const ringStartRadius = 180;
      const ringGap = 150;
      const ringCapacities = [10, 18, 28, 38, 48, 60, 72];
      let remainingNodes = [...subgraph.nodes].sort((left, right) => left.label.localeCompare(right.label));
      let ringIndex = 0;

      while (remainingNodes.length > 0) {{
        const ringCapacity = ringCapacities[ringIndex] || (ringCapacities[ringCapacities.length - 1] + (ringIndex - ringCapacities.length + 1) * 14);
        const radius = ringStartRadius + ringIndex * ringGap;
        const ringNodes = remainingNodes.slice(0, ringCapacity);
        remainingNodes = remainingNodes.slice(ringCapacity);

        ringNodes.forEach((node, indexInRing) => {{
          const angle = (Math.PI * 2 * indexInRing) / ringNodes.length - Math.PI / 2;
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;
          positionedNodes.push({{
            ...node,
            x,
            y,
            lane: "concept-ring",
          }});
        }});
        ringIndex += 1;
      }}

      const positionedMap = new Map(positionedNodes.map(node => [node.id, node]));
      fitSvgToNodes(positionedNodes, 164, 36, 120);

      subgraph.edges.forEach(edge => {{
        const source = positionedMap.get(edge.source);
        const target = positionedMap.get(edge.target);
        if (!source || !target) return;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const startX = source.x + 82;
        const startY = source.y + 18;
        const endX = target.x;
        const endY = target.y + 18;
        const controlX = (startX + endX) / 2;
        const d = `M ${{startX}} ${{startY}} C ${{controlX}} ${{startY}}, ${{controlX}} ${{endY}}, ${{endX}} ${{endY}}`;
        path.setAttribute("d", d);
        path.setAttribute("class", `edge ${{edge.relation}}`);
        if (edge.relation === "concept_prerequisite_of") {{
          path.setAttribute("marker-end", "url(#arrow-concept-prereq)");
        }} else if (edge.relation === "concept_related_to") {{
          path.setAttribute("stroke-dasharray", "6 4");
        }}
        path.addEventListener("mousemove", event => showEdgeTooltip(event, edge));
        path.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(path);
      }});

      positionedNodes.forEach(node => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("class", `node concept${{activeNodeId === node.id ? " active" : ""}}`);
        group.setAttribute("transform", `translate(${{node.x}}, ${{node.y}})`);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("width", "164");
        rect.setAttribute("height", "36");
        group.appendChild(rect);

        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        title.setAttribute("x", "10");
        title.setAttribute("y", "21");
        title.textContent = node.label;
        title.setAttribute("style", "font-size:11px;");
        group.appendChild(title);

        group.addEventListener("click", () => showNodeDetails(node.id));
        group.addEventListener("mousemove", event => showNodeTooltip(event, node));
        group.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(group);
      }});
    }}

    function renderConceptSphere() {{
      const subgraph = getFullConceptSubgraph();
      const conceptNodes = [...subgraph.nodes].sort((left, right) => left.label.localeCompare(right.label));
      svg.setAttribute("viewBox", "0 0 2400 1800");

      // The 3D sphere uses deterministic base positions so the graph feels
      // stable across rerenders, then applies a lightweight rotation in the
      // browser to create the globe effect.
      const baseNodes = conceptNodes.map((node, index) => {{
        const point = fibonacciSpherePoint(index, conceptNodes.length);
        return {{
          ...node,
          x3: point.x,
          y3: point.y,
          z3: point.z,
        }};
      }});

      const renderFrame = () => {{
        svg.innerHTML = "";
        ensureDefs();
        svg.setAttribute("viewBox", "0 0 2400 1800");

        if (!sphereRotationPaused) {{
          sphereRotation += 0.01;
        }}

        const projectedNodes = baseNodes.map(node => projectSphereNode(node, sphereRotation, sphereTilt));
        const projectedMap = new Map(projectedNodes.map(node => [node.id, node]));

        subgraph.edges.forEach(edge => {{
          const source = projectedMap.get(edge.source);
          const target = projectedMap.get(edge.target);
          if (!source || !target) return;

          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          const startX = source.screenX;
          const startY = source.screenY;
          const endX = target.screenX;
          const endY = target.screenY;
          const controlX = (startX + endX) / 2;
          const controlY = (startY + endY) / 2 - Math.abs(source.depth - target.depth) * 120;
          const d = `M ${{startX}} ${{startY}} Q ${{controlX}} ${{controlY}}, ${{endX}} ${{endY}}`;
          const averageDepth = (source.depth + target.depth) / 2;
          const edgeOpacity = 0.12 + (averageDepth + 1) * 0.18;
          path.setAttribute("d", d);
          path.setAttribute("class", `edge ${{edge.relation}}`);
          path.setAttribute("opacity", `${{Math.min(0.45, edgeOpacity)}}`);
          if (edge.relation === "concept_prerequisite_of") {{
            path.setAttribute("marker-end", "url(#arrow-concept-prereq)");
          }} else if (edge.relation === "concept_related_to") {{
            path.setAttribute("stroke-dasharray", "6 4");
          }}
          path.addEventListener("mousemove", event => showEdgeTooltip(event, edge));
          path.addEventListener("mouseleave", hideTooltip);
          svg.appendChild(path);
        }});

        projectedNodes
          .sort((left, right) => left.depth - right.depth)
          .forEach(node => {{
            const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
            const nodeWidth = 92 + Math.min(76, node.label.length * 5);
            const nodeHeight = 28;
            const fontSize = 10 + node.scale * 2.2;
            const rectOpacity = 0.68 + node.scale * 0.28;
            group.setAttribute("class", `node concept${{activeNodeId === node.id ? " active" : ""}}`);
            group.setAttribute("transform", `translate(${{node.screenX - nodeWidth / 2}}, ${{node.screenY - nodeHeight / 2}}) scale(${{node.scale}})`);
            group.setAttribute("opacity", `${{rectOpacity}}`);

            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("width", `${{nodeWidth}}`);
            rect.setAttribute("height", `${{nodeHeight}}`);
            group.appendChild(rect);

            const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
            title.setAttribute("x", "10");
            title.setAttribute("y", "18");
            title.setAttribute("style", `font-size:${{fontSize}}px;`);
            title.textContent = node.label;
            group.appendChild(title);

            group.addEventListener("click", () => showNodeDetails(node.id));
            group.addEventListener("mousemove", event => showNodeTooltip(event, node));
            group.addEventListener("mouseleave", hideTooltip);
            svg.appendChild(group);
          }});

        if (showFullConceptMap.checked && showSphereMode.checked) {{
          sphereAnimationFrame = window.requestAnimationFrame(renderFrame);
        }}
      }};

      renderFrame();
    }}

    function fitSvgToNodes(positionedNodes, nodeWidth, nodeHeight, padding) {{
      if (!positionedNodes.length) {{
        svg.setAttribute("viewBox", defaultViewBox);
        return;
      }}

      // Full concept maps can spill far outside the default focused view, so we
      // recompute the SVG viewBox from the actual node bounds to keep the whole
      // map visible without manual scrolling first.
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;

      positionedNodes.forEach(node => {{
        minX = Math.min(minX, node.x);
        minY = Math.min(minY, node.y);
        maxX = Math.max(maxX, node.x + nodeWidth);
        maxY = Math.max(maxY, node.y + nodeHeight);
      }});

      const viewX = Math.max(0, minX - padding);
      const viewY = Math.max(0, minY - padding);
      const viewWidth = Math.max(600, maxX - minX + padding * 2);
      const viewHeight = Math.max(400, maxY - minY + padding * 2);
      svg.setAttribute("viewBox", `${{viewX}} ${{viewY}} ${{viewWidth}} ${{viewHeight}}`);
    }}

    function fibonacciSpherePoint(index, total) {{
      const offset = 2 / total;
      const increment = Math.PI * (3 - Math.sqrt(5));
      const y = ((index * offset) - 1) + offset / 2;
      const radius = Math.sqrt(1 - y * y);
      const phi = index * increment;
      return {{
        x: Math.cos(phi) * radius,
        y,
        z: Math.sin(phi) * radius,
      }};
    }}

    function projectSphereNode(node, rotationY, tiltX) {{
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);
      const rotatedX = node.x3 * cosY - node.z3 * sinY;
      const rotatedZ = node.x3 * sinY + node.z3 * cosY;

      const cosX = Math.cos(tiltX);
      const sinX = Math.sin(tiltX);
      const tiltedY = node.y3 * cosX - rotatedZ * sinX;
      const tiltedZ = node.y3 * sinX + rotatedZ * cosX;

      const perspective = 2.6;
      const scale = perspective / (perspective - tiltedZ);
      return {{
        ...node,
        screenX: 1200 + rotatedX * 620 * scale,
        screenY: 900 + tiltedY * 620 * scale,
        depth: tiltedZ,
        scale: Math.max(0.52, Math.min(1.2, scale * 0.72)),
      }};
    }}

    function stopSphereAnimation() {{
      if (sphereAnimationFrame !== null) {{
        window.cancelAnimationFrame(sphereAnimationFrame);
        sphereAnimationFrame = null;
      }}
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

    function showNodeTooltip(event, node) {{
      tooltip.style.display = "block";
      tooltip.style.left = `${{event.clientX + 14}}px`;
      tooltip.style.top = `${{event.clientY + 14}}px`;
      tooltip.textContent = node.type === "concept"
        ? `Concept: ${{node.label}}\nId: ${{node.id}}`
        : `Module: ${{node.title}}\nId: ${{node.id}}`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function showNodeDetails(nodeId) {{
      activeNodeId = nodeId;
      nodeSelect.value = nodeId;
      const node = nodeMap.get(nodeId);
      const outgoing = graph.edges.filter(edge => edge.source === nodeId);
      const incoming = graph.edges.filter(edge => edge.target === nodeId);
      const metadata = node.type === "module"
        ? `Title\n${{node.title}}\n\nChapter\n${{node.chapter_title || "Unknown"}}\n\nStart page\n${{node.start_page || "Unknown"}}`
        : `Label\n${{node.label}}\n\nAliases\n${{(node.aliases || []).join(", ") || "None"}}\n\nCourses\n${{(node.courses || []).join(", ") || "None"}}`;

      details.textContent =
`${{node.type === "module" ? "Module" : "Concept"}}
${{node.id}}

${{metadata}}

Outgoing edges
${{outgoing.map(edge => `- ${{edge.relation}} -> ${{edge.target}} (weight=${{edge.weight}}) | ${{edge.reason}}`).join("\\n") || "None"}}

Incoming edges
${{incoming.map(edge => `- ${{edge.relation}} <- ${{edge.source}} (weight=${{edge.weight}}) | ${{edge.reason}}`).join("\\n") || "None"}}`;
      render();
    }}

    Object.values(relationFilters).forEach(checkbox => {{
      checkbox.addEventListener("change", render);
    }});
    showFullConceptMap.addEventListener("change", render);
    showSphereMode.addEventListener("change", render);
    nodeSelect.addEventListener("change", () => showNodeDetails(nodeSelect.value));
    conceptList.addEventListener("click", event => {{
      const button = event.target.closest("[data-node-id]");
      if (!button) return;
      showNodeDetails(button.getAttribute("data-node-id"));
    }});
    fitViewButton.addEventListener("click", render);
    toggleRotateButton.addEventListener("click", () => {{
      sphereRotationPaused = !sphereRotationPaused;
      toggleRotateButton.textContent = sphereRotationPaused ? "resume rotation" : "pause rotation";
      if (!sphereRotationPaused && showFullConceptMap.checked && showSphereMode.checked && sphereAnimationFrame === null) {{
        renderConceptSphere();
      }}
    }});

    showNodeDetails(nodeSelect.value);
  </script>
</body>
</html>
"""


def _serve_graph_view(html_path: Path, port: int, open_browser: bool) -> int:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    with TCPServer(("127.0.0.1", port), Handler) as httpd:
        relative_path = html_path.relative_to(BASE_DIR).as_posix()
        url = f"http://127.0.0.1:{port}/{relative_path}"
        print(f"Serving concept graph viewer at: {url}")

        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nConcept graph viewer stopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
