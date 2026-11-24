#!/usr/bin/env node
"use strict";

// Publication figures for the BDCC major revision. Figures 2 and 3 use only
// the verified cross-dataset synthesis tables. Figure 4 is deliberately gated
// by an approval manifest containing approved=true and exact SHA-256 values for
// both final temporal-contamination summaries and audits.

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const sharp = require("sharp");

const C = {
  cb: "#D55E00", jth: "#0072B2", teal: "#009E73", purple: "#7B61A8",
  dark: "#202124", mid: "#5F6368", grid: "#D9DDE2", axis: "#596068",
  light: "#F7F8FA", zero: "#7B8188", white: "#FFFFFF",
};

function esc(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      if (row.some(v => v !== "")) rows.push(row);
      row = []; field = "";
    } else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  if (!rows.length) return [];
  const header = rows[0];
  return rows.slice(1).map(values => Object.fromEntries(header.map((key, i) => [key, values[i] ?? ""])));
}

function readCsv(file) { return parseCsv(fs.readFileSync(file, "utf8")); }
function num(value) { const out = Number(value); if (!Number.isFinite(out)) throw new Error(`Non-numeric value: ${value}`); return out; }
function sha256(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }
function arg(name, fallback = null) { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : fallback; }
function has(name) { return process.argv.includes(name); }

function svgStart(w, h) {
  return [`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`,
    `<rect width="${w}" height="${h}" fill="#fff"/>`,
    `<style>text{font-family:Arial,Helvetica,sans-serif;fill:${C.dark}} .main{font-size:54px;font-weight:700} .subtitle{font-size:30px;fill:${C.mid}} .panel{font-size:36px;font-weight:700} .axis{font-size:31px} .tick{font-size:28px;fill:${C.mid}} .legend{font-size:27px} .small{font-size:26px;fill:${C.mid}}</style>`];
}

function marker(parts, kind, x, y, color, size = 10, fill = color) {
  if (kind === "circle") parts.push(`<circle cx="${x}" cy="${y}" r="${size}" fill="${fill}" stroke="${color}" stroke-width="3"/>`);
  else if (kind === "square") parts.push(`<rect x="${x - size}" y="${y - size}" width="${2 * size}" height="${2 * size}" fill="${fill}" stroke="${color}" stroke-width="3"/>`);
  else if (kind === "triangle") parts.push(`<path d="M${x} ${y - size - 2} L${x + size + 2} ${y + size} L${x - size - 2} ${y + size} Z" fill="${fill}" stroke="${color}" stroke-width="3"/>`);
  else if (kind === "diamond") parts.push(`<path d="M${x} ${y - size - 2} L${x + size + 2} ${y} L${x} ${y + size + 2} L${x - size - 2} ${y} Z" fill="${fill}" stroke="${color}" stroke-width="3"/>`);
}

function logScale(value, x0, width, min = 5e-4, max = 1) {
  const a = Math.log10(min), b = Math.log10(max);
  return x0 + (Math.log10(value) - a) / (b - a) * width;
}
function linearScale(value, y0, height, min, max) { return y0 + height - (value - min) / (max - min) * height; }

function logTick(parts, x, y, exponent) {
  parts.push(`<text class="tick" x="${x}" y="${y}" text-anchor="middle"><tspan>10</tspan><tspan baseline-shift="super" font-size="20">${exponent}</tspan></text>`);
}

function axes(parts, cfg) {
  const {x, y, w, h, ymin, ymax, yticks, title: panelTitle, xlabel, ylabel} = cfg;
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${C.white}" stroke="${C.axis}" stroke-width="3"/>`);
  for (const value of yticks) {
    const yy = linearScale(value, y, h, ymin, ymax);
    parts.push(`<line x1="${x}" y1="${yy}" x2="${x + w}" y2="${yy}" stroke="${C.grid}" stroke-width="2"/>`);
    parts.push(`<text class="tick" x="${x - 18}" y="${yy + 10}" text-anchor="end">${value.toFixed(1)}</text>`);
  }
  for (const exp of [-3, -2, -1, 0]) {
    const xx = logScale(10 ** exp, x, w);
    parts.push(`<line x1="${xx}" y1="${y}" x2="${xx}" y2="${y + h}" stroke="${C.grid}" stroke-width="2"/>`);
    logTick(parts, xx, y + h + 42, exp);
  }
  parts.push(`<text class="panel" x="${x + w / 2}" y="${y - 25}" text-anchor="middle">${esc(panelTitle)}</text>`);
  parts.push(`<text class="axis" x="${x + w / 2}" y="${y + h + 94}" text-anchor="middle">${esc(xlabel)}</text>`);
  parts.push(`<text class="axis" x="${x - 90}" y="${y + h / 2}" text-anchor="middle" transform="rotate(-90 ${x - 90} ${y + h / 2})">${esc(ylabel)}</text>`);
}

function seriesPath(points, x, y, w, h, ymin, ymax) {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  return sorted.map((p, i) => `${i ? "L" : "M"}${logScale(p.x, x, w).toFixed(2)} ${linearScale(p.y, y, h, ymin, ymax).toFixed(2)}`).join(" ");
}

function drawSeries(parts, points, panel, style) {
  if (!points.length) throw new Error(`Empty series: ${style.label}`);
  const dash = style.dash ? ` stroke-dasharray="${style.dash}"` : "";
  parts.push(`<path d="${seriesPath(points, panel.x, panel.y, panel.w, panel.h, panel.ymin, panel.ymax)}" fill="none" stroke="${style.color}" stroke-width="6"${dash}/>`);
  for (const point of points) marker(parts, style.marker, logScale(point.x, panel.x, panel.w), linearScale(point.y, panel.y, panel.h, panel.ymin, panel.ymax), style.color, 9);
}

function legendItem(parts, x, y, style, label) {
  const dash = style.dash ? ` stroke-dasharray="${style.dash}"` : "";
  parts.push(`<line x1="${x}" y1="${y}" x2="${x + 68}" y2="${y}" stroke="${style.color}" stroke-width="6"${dash}/>`);
  marker(parts, style.marker, x + 34, y, style.color, 9, style.fill ?? style.color);
  parts.push(`<text class="legend" x="${x + 84}" y="${y + 9}">${esc(label)}</text>`);
}

async function writeSvgPng(outDir, stem, parts, w, h) {
  fs.mkdirSync(outDir, {recursive: true});
  const svgPath = path.join(outDir, `${stem}.svg`);
  const pngPath = path.join(outDir, `${stem}.png`);
  const qaPath = path.join(outDir, `qa_${stem}_1200.png`);
  const svg = `${parts.join("")}\n</svg>`;
  fs.writeFileSync(svgPath, svg, "utf8");
  // A 4500 px raster is 600 dpi at the intended 7.5-inch print width. Render
  // from the SVG viewBox, then set the physical-resolution metadata explicitly;
  // asking libvips to rasterize the 2400 px SVG at density=600 first would
  // create a needless 20,000 px intermediate and exceed its safe pixel limit.
  const rendered = sharp(Buffer.from(svg)).resize({width: 4500});
  await rendered.clone().png({compressionLevel: 9}).withMetadata({density: 600}).toFile(pngPath);
  await rendered.clone().resize({width: 1200}).png({compressionLevel: 9}).toFile(qaPath);
  return {svgPath, pngPath, qaPath, width: w, height: h};
}

function metricRows(source) {
  const wanted = new Set(["bm25_full_static", "tfidf_combined_static", "minilm_title_static", "bm25", "tfidf", "semantic"]);
  return source.filter(r => Number(r.k) === 20 && ["hitrate", "map"].includes(r.metric) && wanted.has(r.model)).map(r => ({
    dataset: r.dataset, strategy: r.strategy, metric: r.metric, model: r.model,
    x: num(r.actual_candidate_fraction_mean), y: num(r.signed_bias_sampled_minus_full),
  }));
}

async function figure2(sourceFile, outDir) {
  const rows = metricRows(readCsv(sourceFile));
  if (rows.length !== 120) throw new Error(`Figure 2 expected 120 filtered rows, found ${rows.length}`);
  const W = 2400, H = 1780, p = svgStart(W, H);
  p.push(`<text class="main" x="1200" y="75" text-anchor="middle">Metric bias versus realized candidate fraction</text>`);
  p.push(`<text class="subtitle" x="1200" y="125" text-anchor="middle">Positive values mean that sampled evaluation exceeds the exact-catalog value.</text>`);
  const panels = [
    {metric:"hitrate", strategy:"uniform", title:"HitRate@20 - uniform sampling", x:170, y:245, w:960, h:520, ymin:0, ymax:0.9, yticks:[0,0.2,0.4,0.6,0.8]},
    {metric:"hitrate", strategy:"popularity_weighted", title:"HitRate@20 - popularity-weighted sampling", x:1320, y:245, w:960, h:520, ymin:0, ymax:0.9, yticks:[0,0.2,0.4,0.6,0.8]},
    {metric:"map", strategy:"uniform", title:"MAP@20 - uniform sampling", x:170, y:955, w:960, h:520, ymin:0, ymax:0.65, yticks:[0,0.1,0.2,0.3,0.4,0.5,0.6]},
    {metric:"map", strategy:"popularity_weighted", title:"MAP@20 - popularity-weighted sampling", x:1320, y:955, w:960, h:520, ymin:0, ymax:0.65, yticks:[0,0.1,0.2,0.3,0.4,0.5,0.6]},
  ];
  const styles = [
    {dataset:"CareerBuilder", model:"bm25_full_static", color:C.cb, dash:null, marker:"circle", label:"CareerBuilder BM25 (all text fields)"},
    {dataset:"CareerBuilder", model:"tfidf_combined_static", color:C.cb, dash:"4 10", marker:"square", label:"CareerBuilder combined TF-IDF"},
    {dataset:"CareerBuilder", model:"minilm_title_static", color:C.cb, dash:"16 10", marker:"triangle", label:"CareerBuilder title MiniLM"},
    {dataset:"JTH", model:"bm25", color:C.jth, dash:null, marker:"circle", label:"JTH BM25"},
    {dataset:"JTH", model:"tfidf", color:C.jth, dash:"4 10", marker:"square", label:"JTH TF-IDF"},
    {dataset:"JTH", model:"semantic", color:C.jth, dash:"16 10", marker:"triangle", label:"JTH MiniLM"},
  ];
  for (const panel of panels) {
    axes(p, {...panel, xlabel:"Mean realized candidate fraction (log scale)", ylabel:"Sampled minus exact-catalog metric"});
    p.push(`<line x1="${panel.x}" y1="${linearScale(0,panel.y,panel.h,panel.ymin,panel.ymax)}" x2="${panel.x+panel.w}" y2="${linearScale(0,panel.y,panel.h,panel.ymin,panel.ymax)}" stroke="${C.zero}" stroke-width="3"/>`);
    for (const style of styles) {
      const points = rows.filter(r => r.metric === panel.metric && r.strategy === panel.strategy && r.dataset === style.dataset && r.model === style.model);
      drawSeries(p, points, panel, style);
    }
    marker(p, "diamond", logScale(1,panel.x,panel.w), linearScale(0,panel.y,panel.h,panel.ymin,panel.ymax), C.zero, 10, C.white);
  }
  const lx = [250, 1050, 1815], ly = [1605, 1670];
  styles.forEach((style, i) => legendItem(p, lx[i % 3], ly[Math.floor(i / 3)], style, style.label));
  p.push(`<text class="small" x="1200" y="1740" text-anchor="middle">Open diamonds at fraction 1.0 mark the exact-catalog reference (bias = 0.0000).</text>`);
  return writeSvgPng(outDir, "figure_02_metric_bias_convergence", p, W, H);
}

function aggregateAgreement(rows) {
  const groups = new Map();
  for (const r of rows) {
    const key = [r.dataset,r.strategy,r.candidate_size,r.realized_candidate_fraction_mean].join("|");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  return [...groups.values()].map(group => ({
    dataset: group[0].dataset, strategy: group[0].strategy,
    x: num(group[0].realized_candidate_fraction_mean),
    tau: group.reduce((s,r)=>s+num(r.kendall_tau_b_mean),0)/group.length,
    pairwise: group.reduce((s,r)=>s+num(r.pairwise_order_agreement_mean),0)/group.length,
    top: group.reduce((s,r)=>s+num(r.winner_exact_rate),0)/group.length,
    metrics: group.length,
  }));
}

async function figure3(sourceFile, outDir) {
  const raw = readCsv(sourceFile);
  const data = aggregateAgreement(raw);
  const expectedMetrics = {CareerBuilder:3, JTH:6};
  if (data.length !== 20 || data.some(r => r.metrics !== expectedMetrics[r.dataset])) {
    throw new Error(`Figure 3 expected 20 aggregates with 3 CareerBuilder or 6 JTH metrics; found ${data.length}`);
  }
  const W = 2400, H = 1080, p = svgStart(W, H);
  p.push(`<text class="main" x="1200" y="75" text-anchor="middle">Agreement with exact-catalog model decisions</text>`);
  p.push(`<text class="subtitle" x="1200" y="125" text-anchor="middle">Means are computed from the reported k=20 metrics (CareerBuilder: 3; JTH: 6); exact-catalog agreement is 1.0000.</text>`);
  const panels = [
    {field:"tau", title:"Rank agreement", ylabel:"Mean Kendall tau-b", x:145, y:235, w:650, h:520},
    {field:"pairwise", title:"Pairwise-order agreement", ylabel:"Mean agreement rate", x:885, y:235, w:650, h:520},
    {field:"top", title:"Exact top-model-set agreement", ylabel:"Mean exact-set rate", x:1625, y:235, w:650, h:520},
  ].map(x => ({...x,ymin:0.4,ymax:1.02,yticks:[0.4,0.6,0.8,1.0]}));
  const styles = [
    {dataset:"CareerBuilder", strategy:"uniform", color:C.cb, dash:null, marker:"circle", label:"CareerBuilder uniform"},
    {dataset:"CareerBuilder", strategy:"popularity_weighted", color:C.cb, dash:"16 10", marker:"square", label:"CareerBuilder popularity-weighted"},
    {dataset:"JTH", strategy:"uniform", color:C.jth, dash:null, marker:"circle", label:"JTH uniform"},
    {dataset:"JTH", strategy:"popularity_weighted", color:C.jth, dash:"16 10", marker:"square", label:"JTH popularity-weighted"},
  ];
  for (const panel of panels) {
    axes(p, {...panel, xlabel:"Mean realized candidate fraction (log scale)"});
    const yy = linearScale(1,panel.y,panel.h,panel.ymin,panel.ymax);
    p.push(`<line x1="${panel.x}" y1="${yy}" x2="${panel.x+panel.w}" y2="${yy}" stroke="${C.zero}" stroke-width="3" stroke-dasharray="8 8"/>`);
    for (const style of styles) {
      const points = data.filter(r => r.dataset === style.dataset && r.strategy === style.strategy).map(r => ({x:r.x,y:r[panel.field]}));
      drawSeries(p, points, panel, style);
    }
    marker(p, "diamond", logScale(1,panel.x,panel.w), yy, C.zero, 10, C.white);
  }
  const positions = [[340,900],[1190,900],[340,970],[1190,970]];
  styles.forEach((style,i)=>legendItem(p,positions[i][0],positions[i][1],style,style.label));
  p.push(`<text class="small" x="1200" y="1040" text-anchor="middle">Open diamonds at fraction 1.0 mark exact-catalog agreement (1.0000).</text>`);
  return writeSvgPng(outDir, "figure_03_exact_catalog_decision_agreement", p, W, H);
}

function loadApprovedTemporalManifest(manifestPath) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const manifestDir = path.dirname(path.resolve(manifestPath));
  for (const key of ["approval_evidence", "independent_qa"]) {
    const item = manifest[key];
    if (!item || !item.path || !item.sha256) throw new Error(`Figure 4 manifest missing ${key} evidence`);
    item.path = path.resolve(manifestDir, item.path);
    if (sha256(item.path).toLowerCase() !== item.sha256.toLowerCase()) throw new Error(`Figure 4 ${key} hash mismatch`);
  }
  const authorityText = fs.readFileSync(manifest.approval_evidence.path, "utf8");
  for (const marker of [
    "02_temporal_contamination/jth_full_v2_exact",
    "Revision_R1_FutureContamination_Authoritative_v2",
  ]) {
    if (!authorityText.includes(marker)) throw new Error(`Figure 4 authority register missing approved artifact: ${marker}`);
  }
  const independentQa = JSON.parse(fs.readFileSync(manifest.independent_qa.path, "utf8"));
  if (independentQa.passed !== true || independentQa.require_complete !== true) throw new Error("Figure 4 independent temporal QA did not pass completely");
  for (const dataset of ["careerbuilder", "jth"]) {
    const item = manifest[dataset];
    if (!item || item.approved !== true) throw new Error(`Figure 4 blocked: ${dataset} is not approved`);
    for (const key of ["summary_path", "audit_path", "summary_sha256", "audit_sha256"]) if (!item[key]) throw new Error(`Figure 4 manifest missing ${dataset}.${key}`);
    item.summary_path = path.resolve(manifestDir, item.summary_path);
    item.audit_path = path.resolve(manifestDir, item.audit_path);
    for (const approvedPath of [item.summary_path, item.audit_path]) {
      if (/(^|[\\/_-])(smoke|provisional|superseded|failed)([\\/_-]|$)/i.test(approvedPath)) {
        throw new Error(`Figure 4 blocked: non-final path is not admissible: ${approvedPath}`);
      }
    }
    if (sha256(item.summary_path).toLowerCase() !== item.summary_sha256.toLowerCase()) throw new Error(`${dataset} summary hash mismatch`);
    if (sha256(item.audit_path).toLowerCase() !== item.audit_sha256.toLowerCase()) throw new Error(`${dataset} audit hash mismatch`);
    const qaDataset = independentQa.datasets && independentQa.datasets[dataset];
    if (!qaDataset || qaDataset.status !== "complete") throw new Error(`Figure 4 independent QA is incomplete for ${dataset}`);
    if (Object.values(qaDataset.reference_parity_max_abs || {}).some(value => num(value) !== 0)) throw new Error(`Figure 4 reference parity failed for ${dataset}`);
    const audit = JSON.parse(fs.readFileSync(item.audit_path, "utf8"));
    const declaredSummary = audit.outputs && audit.outputs.summary;
    if (!declaredSummary || String(declaredSummary.sha256).toLowerCase() !== item.summary_sha256.toLowerCase()) {
      throw new Error(`Figure 4 ${dataset} audit does not declare the approved summary hash`);
    }
  }
  return manifest;
}

function temporalChanges(rows, dataset) {
  const isCb = dataset === "careerbuilder";
  const ref = isCb ? "cutoff_active_reference" : "created_by_query_date";
  const contaminated = isCb ? "future_opening_contaminated" : "future_contaminated_all_jobs";
  const filtered = rows.filter(r => Number(r.k) === 20 && (!isCb || r.estimand === "design_weighted_population") && (isCb || r.group === "all"));
  const keyed = new Map(filtered.map(r => [[r.model,r.catalog_regime].join("|"),r]));
  const modelOrder = isCb
    ? ["bm25_full_static","tfidf_combined_static","popularity_precut","bm25_title_static","minilm_title_static","random_hash"]
    : ["bm25","tfidf","semantic","recency_pop","item_knn","random","popularity"];
  const present = new Set(filtered.map(r=>r.model));
  if (present.size !== modelOrder.length || modelOrder.some(model => !present.has(model))) {
    throw new Error(`${dataset}: temporal model set differs from the locked ${modelOrder.length}-model design`);
  }
  const models = modelOrder;
  return models.map(model => {
    const a = keyed.get(`${model}|${ref}`), b = keyed.get(`${model}|${contaminated}`);
    if (!a || !b) throw new Error(`${dataset}: incomplete temporal pair for ${model}`);
    return {model, hit_rate:num(b.hit_rate)-num(a.hit_rate), map:num(b.map)-num(a.map), reference_candidates:num(a.candidate_count_mean), future_added:num(b.future_jobs_added_mean)};
  });
}

function temporalModelLabel(dataset, model) {
  const cb={random_hash:"Random",popularity_precut:"Popularity",tfidf_combined_static:"TF-IDF",bm25_full_static:"BM25 (all text fields)",bm25_title_static:"Title BM25",minilm_title_static:"Title MiniLM"};
  const jth={random:"Random",popularity:"Popularity",tfidf:"TF-IDF",bm25:"BM25",recency_pop:"Recency-popularity",item_knn:"Item-kNN",semantic:"MiniLM"};
  return (dataset === "careerbuilder" ? cb : jth)[model] || model;
}

function temporalPanel(parts, x, y, w, h, title, rows, dataset, limit) {
  const xscale = v => x + (v + limit)/(2*limit)*w;
  const rowGap = h/(rows.length+1);
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#fff" stroke="${C.axis}" stroke-width="3"/>`);
  parts.push(`<text class="panel" x="${x+w/2}" y="${y-78}" text-anchor="middle">${esc(title)}</text>`);
  const refMean=rows.reduce((s,r)=>s+r.reference_candidates,0)/rows.length, futMean=rows.reduce((s,r)=>s+r.future_added,0)/rows.length;
  const countFormat=new Intl.NumberFormat("en-US",{minimumFractionDigits:1,maximumFractionDigits:1});
  parts.push(`<text class="small" x="${x+w/2}" y="${y-40}" text-anchor="middle">Reference catalog: ${countFormat.format(refMean)} mean candidates</text>`);
  parts.push(`<text class="small" x="${x+w/2}" y="${y-10}" text-anchor="middle">Future-inclusive catalog: +${countFormat.format(futMean)} jobs (${(100*futMean/refMean).toFixed(2)}%)</text>`);
  for(const tick of [-limit,-limit/2,0,limit/2,limit]){const xx=xscale(tick);parts.push(`<line x1="${xx}" y1="${y}" x2="${xx}" y2="${y+h}" stroke="${tick===0?C.zero:C.grid}" stroke-width="${tick===0?4:2}"/>`);parts.push(`<text class="tick" x="${xx}" y="${y+h+38}" text-anchor="middle">${tick.toFixed(4)}</text>`)}
  rows.forEach((r,i)=>{const yy=y+rowGap*(i+1);const xHit=xscale(r.hit_rate),xMap=xscale(r.map);parts.push(`<text class="tick" x="${x-18}" y="${yy+9}" text-anchor="end">${esc(temporalModelLabel(dataset,r.model))}</text>`);parts.push(`<line x1="${x}" y1="${yy}" x2="${x+w}" y2="${yy}" stroke="${C.grid}" stroke-width="1"/>`);parts.push(`<line x1="${Math.min(xHit,xMap)}" y1="${yy}" x2="${Math.max(xHit,xMap)}" y2="${yy}" stroke="${C.zero}" stroke-width="3"/>`);marker(parts,"circle",xHit,yy,C.cb,10);marker(parts,"square",xMap,yy,C.jth,10,"none");});
  parts.push(`<text class="axis" x="${x+w/2}" y="${y+h+88}" text-anchor="middle">Future-inclusive minus reference-catalog metric</text>`);
}

async function figure4(manifest, outDir) {
  const cb=temporalChanges(readCsv(manifest.careerbuilder.summary_path),"careerbuilder");
  const jth=temporalChanges(readCsv(manifest.jth.summary_path),"jth");
  const maxAbs=Math.max(...[...cb,...jth].flatMap(r=>[Math.abs(r.hit_rate),Math.abs(r.map)]));
  const sharedLimit=Math.max(0.005,Math.ceil((maxAbs*1.10)/0.005)*0.005);
  const W=2400,H=1220,p=svgStart(W,H);
  p.push(`<text class="main" x="1200" y="75" text-anchor="middle">Sensitivity to including future-opening jobs</text>`);
  p.push(`<text class="subtitle" x="1200" y="125" text-anchor="middle">Future-inclusive catalog minus reference catalog; queries, targets, shared-job scores, metrics, and tie rules are fixed.</text>`);
  temporalPanel(p,330,260,760,650,"CareerBuilder",cb,"careerbuilder",sharedLimit);
  temporalPanel(p,1440,260,760,650,"JTH",jth,"jth",sharedLimit);
  legendItem(p,760,1060,{color:C.cb,dash:null,marker:"circle"},"HitRate@20 difference");
  legendItem(p,1330,1060,{color:C.jth,dash:null,marker:"square",fill:"none"},"MAP@20 difference");
  p.push(`<text class="small" x="1200" y="1140" text-anchor="middle">Negative values indicate lower logged-reference effectiveness after future-opening jobs are added.</text>`);
  return writeSvgPng(outDir,"figure_04_future_job_eligibility_sensitivity",p,W,H);
}

async function main() {
  const root=path.resolve(__dirname,"..");
  const project=path.resolve(root,"..","..");
  const outDir=path.resolve(arg("--output-dir",path.join(project,"Manuscript","Revision_R1_BDCC-4711631_2026-11-23","03_Working_Manuscript","figures_draft")));
  const metricFile=path.resolve(arg("--metric-bias",path.join(project,"Results","CrossDataset_EvaluationValidity_Synthesis_v1","table_2_cross_dataset_metric_bias.csv")));
  const agreementFile=path.resolve(arg("--decision-agreement",path.join(project,"Results","CrossDataset_EvaluationValidity_Synthesis_v1","table_3_cross_dataset_rank_fidelity.csv")));
  const outputs=[await figure2(metricFile,outDir),await figure3(agreementFile,outDir)];
  let temporalManifestPath=null, temporalManifest=null;
  if(has("--temporal-manifest")) {
    temporalManifestPath=path.resolve(arg("--temporal-manifest"));
    temporalManifest=loadApprovedTemporalManifest(temporalManifestPath);
    outputs.push(await figure4(temporalManifest,outDir));
  }
  const notePath=path.join(outDir,"REVISION_DATA_FIGURE_PROVENANCE.md");
  const lines=[
    "# Revision data-figure provenance", "", `Generated: ${new Date().toISOString()}`, "",
    "## Verified inputs", "",
    `- Metric-bias table: \`${metricFile}\`; SHA-256 \`${sha256(metricFile)}\``,
    `- Decision-agreement table: \`${agreementFile}\`; SHA-256 \`${sha256(agreementFile)}\``,
    "", "## Transformations", "",
    "- Figure 2 retains k=20 HitRate and MAP rows for BM25, TF-IDF, and MiniLM on each dataset under uniform and popularity-weighted sampling. Curves use source-precision values; the open diamond at fraction 1.0 is the analytical exact-catalog reference (bias 0.0000).",
    "- Figure 3 averages Kendall tau-b, pairwise-order agreement, and exact top-model-set agreement within each dataset, sampler, and candidate size. CareerBuilder contributes its three reported k=20 metrics; JTH contributes its six reported k=20 metrics. Exact-catalog agreement is 1.0000.",
    "- Dataset is encoded by color; model or sampler is also encoded by line style and marker shape so color is not the sole cue.",
    "- The figures use exact-catalog agreement terminology and avoid ambiguous shorthand for the reference condition.",
    "", "## Generator", "",
    `- \`${__filename}\`; SHA-256 \`${sha256(__filename)}\``,
    "",
  ];
  if(temporalManifest){
    lines.push("## Approved Figure 4 inputs", "");
    lines.push(`- Approval manifest: \`${temporalManifestPath}\`; SHA-256 \`${sha256(temporalManifestPath)}\``);
    lines.push(`- Authority register: \`${temporalManifest.approval_evidence.path}\`; SHA-256 \`${sha256(temporalManifest.approval_evidence.path)}\``);
    lines.push(`- Independent fail-closed QA: \`${temporalManifest.independent_qa.path}\`; SHA-256 \`${sha256(temporalManifest.independent_qa.path)}\``);
    for(const dataset of ["careerbuilder","jth"]){const item=temporalManifest[dataset];lines.push(`- ${dataset} summary: \`${item.summary_path}\`; SHA-256 \`${sha256(item.summary_path)}\``);lines.push(`- ${dataset} audit: \`${item.audit_path}\`; SHA-256 \`${sha256(item.audit_path)}\``);}
    lines.push("", "- Figure 4 reports paired HitRate@20 and MAP@20 differences between each protocol-defined reference catalog and a future-inclusive catalog formed by adding future-opening jobs. It includes the locked six CareerBuilder models and seven JTH models; RRF is outside this temporal experiment.", "- Both panels share one symmetric axis. Color, marker shape, and the within-row connector jointly encode the two metrics.", "");
  }
  lines.push("## Outputs", "");
  for(const o of outputs){
    lines.push(`- \`${o.svgPath}\`; SHA-256 \`${sha256(o.svgPath)}\``);
    lines.push(`- \`${o.pngPath}\`; SHA-256 \`${sha256(o.pngPath)}\``);
    lines.push(`- QA thumbnail: \`${o.qaPath}\``);
  }
  lines.push("","## Figure 4 gate","",temporalManifest ? "Gate passed for this rendering: authority register, independent complete QA, exact reference parity, dataset audit declarations, model grids, and all supplied SHA-256 values were validated." : "Figure 4 is generated only when `--temporal-manifest` is supplied. The manifest must mark both datasets `approved: true` and provide authority, independent-QA, summary, and audit files with matching SHA-256 values. Paths marked smoke, superseded, failed, or provisional are rejected.","");
  fs.writeFileSync(notePath,lines.join("\n"),"utf8");
  process.stdout.write(`${JSON.stringify({outputs,notePath},null,2)}\n`);
}

main().catch(error=>{console.error(error.stack||error);process.exitCode=1;});
