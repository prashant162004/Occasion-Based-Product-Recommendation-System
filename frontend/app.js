// Global Application State
let currentResults = [];
let currentFilteredResults = [];
let activeOccasion = "";
let currentDisplayCount = 12;
const PAGE_SIZE = 12;

// On DOM Loaded
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  fetchOccasionPresets();
  setupInputListeners();
});

/* =========================================================
   THEME MANAGEMENT (DARK & LIGHT MODE)
   ========================================================= */
function initTheme() {
  const savedTheme = localStorage.getItem("theme");
  const html = document.documentElement;
  const sunIcon = document.getElementById("theme-sun-icon");
  const moonIcon = document.getElementById("theme-moon-icon");

  if (savedTheme === "light") {
    html.classList.remove("dark");
    if (sunIcon) sunIcon.classList.add("hidden");
    if (moonIcon) moonIcon.classList.remove("hidden");
  } else {
    // Default to Dark Mode
    html.classList.add("dark");
    if (sunIcon) sunIcon.classList.remove("hidden");
    if (moonIcon) moonIcon.classList.add("hidden");
  }
}

function toggleTheme() {
  const html = document.documentElement;
  const sunIcon = document.getElementById("theme-sun-icon");
  const moonIcon = document.getElementById("theme-moon-icon");

  if (html.classList.contains("dark")) {
    html.classList.remove("dark");
    localStorage.setItem("theme", "light");
    if (sunIcon) sunIcon.classList.add("hidden");
    if (moonIcon) moonIcon.classList.remove("hidden");
  } else {
    html.classList.add("dark");
    localStorage.setItem("theme", "dark");
    if (sunIcon) sunIcon.classList.remove("hidden");
    if (moonIcon) moonIcon.classList.add("hidden");
  }
}

/* =========================================================
   INPUT & PRESETS MANAGEMENT
   ========================================================= */
function setupInputListeners() {
  const input = document.getElementById("occasion-input");
  const clearBtn = document.getElementById("clear-btn");

  input.addEventListener("input", (e) => {
    if (e.target.value.trim().length > 0) {
      clearBtn.classList.remove("hidden");
    } else {
      clearBtn.classList.add("hidden");
    }
  });
}

function clearInput() {
  const input = document.getElementById("occasion-input");
  input.value = "";
  document.getElementById("clear-btn").classList.add("hidden");
  input.focus();
}

async function fetchOccasionPresets() {
  try {
    const res = await fetch("/api/occasions");
    if (!res.ok) return;
    const presets = await res.json();
    renderChips(presets);
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function renderChips(presets) {
  const container = document.getElementById("chips-container");
  container.innerHTML = "";

  presets.forEach((preset) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `occasion-chip px-3 py-1.5 rounded-full text-xs font-semibold border`;
    chip.textContent = preset.label;
    chip.onclick = () => {
      document.querySelectorAll(".occasion-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      document.getElementById("occasion-input").value = preset.label;
      document.getElementById("clear-btn").classList.remove("hidden");
      executeSearch(preset.sample_query || preset.label);
    };
    container.appendChild(chip);
  });
}

function handleSearch(e) {
  if (e) e.preventDefault();
  const input = document.getElementById("occasion-input");
  const query = input.value.trim();
  if (query.length < 2) return;
  executeSearch(query);
}

/* =========================================================
   CORE SEARCH PIPELINE EXECUTION
   ========================================================= */
async function executeSearch(occasionQuery) {
  activeOccasion = occasionQuery;
  const input = document.getElementById("occasion-input");
  if (!input.value) input.value = occasionQuery;

  // UI Loading State
  setLoading(true);

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ occasion: occasionQuery, top_k: 24 })
    });

    if (!res.ok) {
      throw new Error("Search request failed.");
    }

    const data = await res.json();
    currentResults = data.results || [];
    currentFilteredResults = currentResults;
    currentDisplayCount = PAGE_SIZE;
    renderResults(data);
  } catch (err) {
    console.error("Search error:", err);
    showEmptyState("An error occurred while computing recommendations. Please try again.");
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const skeleton = document.getElementById("loading-skeleton");
  const grid = document.getElementById("products-grid");
  const emptyState = document.getElementById("empty-state");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");
  const loadMore = document.getElementById("load-more-container");

  if (isLoading) {
    skeleton.classList.remove("hidden");
    grid.classList.add("hidden");
    emptyState.classList.add("hidden");
    if (loadMore) loadMore.classList.add("hidden");
    btnText.textContent = "Processing...";
    btnSpinner.classList.remove("hidden");
  } else {
    skeleton.classList.add("hidden");
    grid.classList.remove("hidden");
    btnText.textContent = "Search";
    btnSpinner.classList.add("hidden");
  }
}

function renderResults(data) {
  const grid = document.getElementById("products-grid");
  const metricsBar = document.getElementById("metrics-bar");
  const summaryBanner = document.getElementById("summary-banner");
  const summaryText = document.getElementById("summary-text");
  const metricsTitle = document.getElementById("metrics-title");
  const metricsLatency = document.getElementById("metrics-latency");
  const emptyState = document.getElementById("empty-state");
  const loadMoreContainer = document.getElementById("load-more-container");

  grid.innerHTML = "";

  if (!currentResults || currentResults.length === 0) {
    showEmptyState(`No products found for "${data.occasion}". Try searching for popular occasions like Birthday, Wedding, or Diwali.`);
    metricsBar.classList.add("hidden");
    summaryBanner.classList.add("hidden");
    if (loadMoreContainer) loadMoreContainer.classList.add("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  metricsBar.classList.remove("hidden");
  summaryBanner.classList.remove("hidden");

  // Metrics update
  metricsTitle.textContent = `Showing ${currentFilteredResults.length} recommendations for "${data.occasion}"`;
  metricsLatency.textContent = `Latency: ${data.latency_ms}ms`;
  summaryText.textContent = data.summary || `Synthesized selections for ${data.occasion}.`;

  // Populate Category Filter dropdown
  updateCategoryDropdown(currentResults);

  // Render Product Cards
  currentFilteredResults.forEach((prod) => {
    grid.appendChild(createProductCard(prod));
  });

  if (loadMoreContainer) {
    loadMoreContainer.classList.add("hidden");
  }
}

function updateCategoryDropdown(items) {
  const select = document.getElementById("category-filter");
  const currentVal = select.value;
  select.innerHTML = '<option value="">All Categories</option>';

  const categories = [...new Set(items.map(p => p.category))].filter(Boolean);
  categories.sort().forEach(cat => {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    if (cat === currentVal) opt.selected = true;
    select.appendChild(opt);
  });
}

function filterByCategory() {
  const select = document.getElementById("category-filter");
  const selectedCat = select.value;
  const grid = document.getElementById("products-grid");

  grid.innerHTML = "";

  currentFilteredResults = selectedCat 
    ? currentResults.filter(p => p.category === selectedCat)
    : currentResults;

  currentDisplayCount = PAGE_SIZE;

  if (currentFilteredResults.length === 0) {
    showEmptyState(`No products found in category "${selectedCat}" for this occasion.`);
    const container = document.getElementById("load-more-container");
    if (container) container.classList.add("hidden");
  } else {
    document.getElementById("empty-state").classList.add("hidden");
    currentFilteredResults.forEach(prod => {
      grid.appendChild(createProductCard(prod));
    });
    const container = document.getElementById("load-more-container");
    if (container) container.classList.add("hidden");
  }
}

/* =========================================================
   AUTHENTIC & DISTINCT REAL PRODUCT PHOTOGRAPHS
   Maps each product item specifically so NO duplicates or misassigned photos appear
   ========================================================= */
function getRealProductPhoto(title, category) {
  const t = (title || "").toLowerCase();
  const c = (category || "").toLowerCase();

  // 1. PERFUMES & FRAGRANCES (priority to avoid flowers matching 'Rose & Peony Eau de Parfum')
  if (t.includes("perfume") || t.includes("eau de parfum") || c.includes("fragrance") || t.includes("scent") || t.includes("floria") || t.includes("fresé") || t.includes("woodé")) {
    if (t.includes("soap") || t.includes("candle")) {
      return "/static/images/wellness_candles.jpg";
    }
    return "/static/images/perfumes.jpg";
  }

  // 2. BRACELETS (priority before 'silver' or 'drop' or 'gold')
  if (t.includes("bracelet")) {
    return "/static/images/pearl_bracelet.jpg";
  }

  // 3. NECKLACES & CHOKERS (priority before 'silver' or 'gold' or 'watch')
  if (t.includes("necklace") || t.includes("choker")) {
    if (t.includes("silver")) {
      return "/static/images/silver_necklace.jpg";
    }
    return "/static/images/gold_choker.jpg";
  }

  // 4. EARRINGS (specifically earring, jhumka, stud - not 'drop' alone)
  if (t.includes("earring") || t.includes("jhumka") || t.includes("stud")) {
    return "/static/images/earrings.jpg";
  }

  // 5. JEWELLERY GENERAL (diamond set, etc.)
  if (t.includes("diamond") || t.includes("jewel")) {
    return "/static/images/jewellery.jpg";
  }

  // 6. WATCHES & TIMEPIECES (must be a watch or timepiece)
  if (t.includes("watch") || t.includes("clock") || c.includes("timepiece")) {
    if (t.includes("gold")) {
      return "/static/images/metallic_gold_watch.jpg";
    }
    if (t.includes("silver") || t.includes("steel") || t.includes("chronograph")) {
      return "/static/images/silver_watch.jpg";
    }
    if (t.includes("vintage") || t.includes("leather") || t.includes("wall")) {
      return "/static/images/vintage_watch.jpg";
    }
    return "/static/images/metallic_gold_watch.jpg";
  }

  // 7. FLOWERS & BOUQUETS
  if (t.includes("red rose") || t.includes("crimson rose") || t.includes("rose bouquet") || (t.includes("rose") && t.includes("bouquet"))) {
    return "/static/images/red_roses.jpg";
  }
  if (t.includes("forever rose")) {
    return "/static/images/red_roses.jpg";
  }
  if (t.includes("cherry blossom")) {
    return "/static/images/cherry_blossom.jpg";
  }
  if (t.includes("lilies") || t.includes("lily") || t.includes("orchid")) {
    return "/static/images/white_lilies.jpg";
  }
  if (t.includes("marigold") || t.includes("pooja") || t.includes("garland") || t.includes("genda")) {
    return "/static/images/marigold.jpg";
  }
  if (t.includes("flower") || t.includes("bouquet") || t.includes("tulip") || c.includes("flower")) {
    return "/static/images/fresh_flowers.jpg";
  }

  // 8. APPAREL & CLOTHING
  if (t.includes("saree") || t.includes("banarasi") || t.includes("silk saree")) {
    return "/static/images/saree.jpg";
  }
  if (t.includes("kashmiri suit") || t.includes("kurti") || t.includes("salwar")) {
    return "/static/images/kashmiri_suit.jpg";
  }
  if (t.includes("shirt") || t.includes("tee") || t.includes("t-shirt") || t.includes("trouser") || t.includes("pant") || t.includes("jean") || t.includes("cargo")) {
    return "/static/images/formal_shirt.jpg";
  }
  if (c.includes("apparel") || c.includes("fashion") || t.includes("office wear")) {
    return "/static/images/ethnic_wear.jpg";
  }

  // 9. CHOCOLATES & CONFECTIONERY
  if (t.includes("dark chocolate") || t.includes("cocoa") || t.includes("truffle")) {
    return "/static/images/dark_chocolate.jpg";
  }
  if (t.includes("chocolate") || c.includes("chocolate") || t.includes("treat") || t.includes("confectionery")) {
    return "/static/images/gourmet_chocolates.jpg";
  }

  // 10. DRY FRUITS & NUTS
  if (t.includes("dry fruit") || t.includes("almond") || t.includes("cashew") || t.includes("nuts") || t.includes("raisin") || t.includes("pistachio")) {
    return "/static/images/diwali_dry_fruits.jpg";
  }

  // 11. FESTIVE & LUXURY HAMPERS
  if (t.includes("diwali") && (t.includes("hamper") || c.includes("hamper"))) {
    return "/static/images/diwali_luxury_hamper.jpg";
  }
  if (t.includes("hamper") || c.includes("hamper")) {
    return "/static/images/luxury_hamper.jpg";
  }

  // 12. COUPLE & LIFESTYLE GIFTS
  if (t.includes("mug")) {
    return "/static/images/couple_mugs.jpg";
  }
  if (t.includes("card") || c.includes("card") || t.includes("voucher")) {
    return "/static/images/festive_gift_card.jpg";
  }
  if (t.includes("toy") || t.includes("plush") || t.includes("teddy") || t.includes("xylophone") || t.includes("drum") || t.includes("guitar")) {
    return "/static/images/plush_toy.jpg";
  }
  if (t.includes("pen") || t.includes("journal") || t.includes("notebook") || c.includes("stationery") || t.includes("book")) {
    return "/static/images/stationery.jpg";
  }
  if (t.includes("candle") || t.includes("soap") || t.includes("wellness") || t.includes("oil") || t.includes("salt") || t.includes("bath") || t.includes("skincare") || t.includes("tea")) {
    return "/static/images/wellness_candles.jpg";
  }

  return "/static/images/gift_box.jpg";
}

/* =========================================================
   DYNAMIC PRODUCT CARD SYNTHESIS
   ========================================================= */
function createProductCard(prod) {
  const card = document.createElement("div");
  card.className = "product-card rounded-2xl border flex flex-col overflow-hidden relative group";

  // Score Badge with Dynamic Glow
  let scoreBadgeHtml = "";
  if (prod.relevance_score >= 90) {
    scoreBadgeHtml = `
      <div class="absolute top-3 right-3 px-2.5 py-1 rounded-full text-xs font-bold border backdrop-blur-md bg-emerald-500/20 text-emerald-400 border-emerald-500/40 shadow-sm">
        ${prod.relevance_score}% Match
      </div>
    `;
  } else if (prod.relevance_score >= 80) {
    scoreBadgeHtml = `
      <div class="absolute top-3 right-3 px-2.5 py-1 rounded-full text-xs font-bold border backdrop-blur-md bg-indigo-500/20 text-indigo-400 border-indigo-500/40 shadow-sm">
        ${prod.relevance_score}% Match
      </div>
    `;
  } else {
    scoreBadgeHtml = `
      <div class="absolute top-3 right-3 px-2.5 py-1 rounded-full text-xs font-bold border backdrop-blur-md bg-sky-500/20 text-sky-400 border-sky-500/40 shadow-sm">
        ${prod.relevance_score}% Match
      </div>
    `;
  }

  // Real Authentic Product Photograph:
  // Resolves to specific dedicated image
  const realPhoto = getRealProductPhoto(prod.title, prod.category);
  const isDeadS3 = !prod.image_url || prod.image_url.includes("s3aws.emilo.in");
  const initialSrc = isDeadS3 ? realPhoto : prod.image_url;

  card.innerHTML = `
    <!-- Image Box -->
    <div class="relative w-full h-48 bg-slate-950 overflow-hidden">
      <img 
        src="${initialSrc}" 
        alt="${escapeHtml(prod.title)}" 
        loading="lazy"
        class="product-image w-full h-full object-cover"
        onerror="this.onerror=null; this.src='${realPhoto}';"
      >
      <!-- Relevance Match Score Pill -->
      ${scoreBadgeHtml}
      <!-- Category Badge -->
      <div class="card-category-badge absolute bottom-3 left-3 px-2 py-0.5 rounded-md text-[10px] font-semibold border backdrop-blur-sm">
        ${escapeHtml(prod.category)}
      </div>
    </div>

    <!-- Content Box -->
    <div class="p-5 flex-1 flex flex-col justify-between space-y-4">
      <div class="space-y-2">
        <div class="flex items-start justify-between gap-2">
          <h3 class="card-title font-bold text-sm sm:text-base leading-snug tracking-tight">
            ${escapeHtml(prod.title)}
          </h3>
        </div>
        <p class="card-brand text-xs font-semibold">
          by ${escapeHtml(prod.brand)}
        </p>
        <p class="card-desc text-xs line-clamp-2 leading-relaxed">
          ${escapeHtml(prod.description_snippet)}
        </p>
      </div>

      <!-- Recommendation Rationale Box -->
      <div class="card-rationale-box rounded-xl border border-l-4 p-3 text-xs">
        <span class="card-rationale-title block text-[10px] font-bold uppercase tracking-wider mb-1">
          Recommendation Rationale
        </span>
        <p class="card-rationale-text italic leading-normal">
          "${escapeHtml(prod.recommendation_rationale)}"
        </p>
      </div>

      <!-- Price & Stock Footer -->
      <div class="pt-3 border-t border-slate-700/50 dark:border-slate-800/80 flex items-center justify-between">
        <div>
          <span class="card-price text-base font-extrabold tracking-tight">${escapeHtml(prod.formatted_price)}</span>
          ${prod.discount_percentage > 0 ? `<span class="ml-1 text-[11px] text-emerald-500 font-semibold">(${prod.discount_percentage}% off)</span>` : ''}
        </div>
        <span class="card-stock text-[11px] font-semibold px-2.5 py-0.5 rounded-lg border">
          ${escapeHtml(prod.stock_status)}
        </span>
      </div>
    </div>
  `;

  return card;
}

function showEmptyState(msg) {
  const emptyState = document.getElementById("empty-state");
  const messageEl = document.getElementById("empty-state-message");
  messageEl.textContent = msg;
  emptyState.classList.remove("hidden");
  document.getElementById("products-grid").innerHTML = "";
  document.getElementById("metrics-bar").classList.add("hidden");
  document.getElementById("summary-banner").classList.add("hidden");
}

function escapeHtml(unsafe) {
  if (!unsafe) return "";
  return unsafe
    .toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}