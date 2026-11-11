import json
import logging
import os
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger("LLMEngine")

OCCASION_KNOWLEDGE_BASE = {
    "wedding": {
        "intent": "Formal marriage ceremonies, bridal celebrations, and wedding gifts",
        "keywords": ["luxury watch", "ethnic suit", "saree", "jewellery", "fragrance", "gift hamper", "gold", "choker"],
        "summary": "Shortlisted premium celebration gifts, luxury timepieces, and ethnic formal attire suitable for wedding ceremonies and receptions."
    },
    "birthday": {
        "intent": "Personal birthday celebrations, milestone ages, and heartfelt gifts",
        "keywords": ["chocolates", "perfume", "flowers", "toys", "watch", "greeting cards", "diy kits"],
        "summary": "Curated birthday gift selections ranging from artisanal treats and floral bouquets to timeless accessories."
    },
    "diwali": {
        "intent": "Auspicious festive celebrations, prosperity gifts, and family gatherings",
        "keywords": ["dry fruits hamper", "marigold flowers", "scented candles", "luxury gift box", "ethnic wear", "kashmiri suits"],
        "summary": "Curated festive assortments focusing on traditional dry fruit hampers, handcrafted decor, and celebration essentials."
    },
    "valentine": {
        "intent": "Romantic celebrations, couples, anniversaries, and date nights",
        "keywords": ["forever rose", "couple perfumes", "luxury chocolates", "touch bracelets", "scented candles", "rose gold necklace"],
        "summary": "Selected romantic keepsakes, couples fragrance sets, and elegant jewelry crafted for romantic expressions."
    },
    "office": {
        "intent": "Corporate milestones, team appreciation, office parties, and professional gifts",
        "keywords": ["executive pens", "formal shirts", "self-care corporate kits", "notebooks", "wellness gifts", "digital gift cards"],
        "summary": "Curated professional corporate gift items focusing on executive stationery, formal office wear, and employee wellness kits."
    },
    "housewarming": {
        "intent": "New home celebrations, interior decoration, and blessing gifts",
        "keywords": ["plant based candles", "natural fiber decor", "indoor plants", "recycled gifts", "greeting cards"],
        "summary": "Shortlisted welcoming home decor, natural living essentials, and indoor botanical gifts for new homeowners."
    },
    "baby shower": {
        "intent": "Welcoming newborn, expectant parents, and maternity celebrations",
        "keywords": ["baby shower gifts", "soft plushies", "teddy bears", "greeting cards", "diy kits"],
        "summary": "Selected gentle newborn essentials, soft cuddly plush toys, and celebratory keepsakes for parents."
    }
}

class LLMEngine:
    """
    Intelligent Occasion Synthesis & Reasoning Service.
    Expands user intent and synthesizes context-aware recommendation rationales.
    Features automated resilient fallback when external API credentials are not set.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", ""))
        self.model_name = model_name
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key, transport="rest")
                self.client = genai.GenerativeModel(self.model_name)
                logger.info(f"[INFO] External AI reasoning provider initialized ({self.model_name}).")
            except Exception as e:
                logger.warning(f"[WARNING] External AI provider initialization notice ({e}). Operating in deterministic engine mode.")
                self.client = None
        else:
            logger.info("[INFO] No external API key provided. Using built-in deterministic reasoning engine.")

    def expand_occasion_intent(self, occasion_query: str) -> Dict[str, Any]:
        """Expands raw user query into enriched semantic attributes using Gemini LLM with robust fallback."""
        if self.client is not None:
            try:
                prompt = (
                    f"You are an expert AI e-commerce gifting strategist.\n"
                    f"Occasion: '{occasion_query}'\n\n"
                    f"Task: Analyze the recipient, tone, and theme of this occasion. Respond in strict valid JSON format:\n"
                    f"{{\n"
                    f'  "intent": "1 sentence describing the purpose, sentiment, and relationship context",\n'
                    f'  "search_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],\n'
                    f'  "summary": "1 sentence summarizing the curated gifting strategy for this occasion"\n'
                    f"}}\n"
                    f"Rules:\n"
                    f"- Do NOT use markdown code blocks or backticks.\n"
                    f"- Do NOT use any emojis.\n"
                    f"- Keep keywords relevant to physical e-commerce gifts (watches, hampers, apparel, jewelry, fragrances, chocolates, decor).\n"
                )
                response = self.client.generate_content(prompt, request_options={"timeout": 4.0})
                raw_text = response.text.strip()
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    intent = parsed.get("intent", f"Gifting celebration for {occasion_query}")
                    kws = parsed.get("search_keywords", [])
                    kw_str = " ".join(kws[:5]) if isinstance(kws, list) else str(kws)
                    summary = parsed.get("summary", f"Curated gift recommendations tailored for {occasion_query}.")
                    logger.info(f"[INFO] Dynamic LLM intent expansion successful for '{occasion_query}'.")
                    return {
                        "detected_intent": intent,
                        "search_query": f"{occasion_query} {kw_str}".strip(),
                        "summary": summary
                    }
            except Exception as e:
                logger.warning(f"[WARNING] Dynamic LLM expansion notice: {e}. Falling back to semantic engine.")

        q_clean = occasion_query.lower()

        # Check knowledge base matches
        for key, data in OCCASION_KNOWLEDGE_BASE.items():
            if key in q_clean:
                return {
                    "detected_intent": data["intent"],
                    "search_query": f"{occasion_query} {' '.join(data['keywords'][:4])}",
                    "summary": data["summary"]
                }

        # Fallback expansion
        return {
            "detected_intent": f"Celebration and gifting for: {occasion_query}",
            "search_query": f"{occasion_query} gift collection premium curated",
            "summary": f"Curated product recommendations tailored for {occasion_query}."
        }

    def filter_and_rationalize(self, occasion: str, candidate_products: List[Dict[str, Any]], max_items: int = 12) -> List[Dict[str, Any]]:
        """
        Dynamically filters candidates using AI reasoning to eliminate conflicting festival/occasion items
        (e.g., removing Diwali/Holi hampers from Wedding searches), and synthesizes context-aware rationales.
        """
        if not candidate_products:
            return []

        # Step 1: Pre-filter conflicting occasion/festival items
        conflict_map = {
            "wedding": ["diwali", "holi", "eid", "christmas", "baby shower", "newborn", "halloween"],
            "marriage": ["diwali", "holi", "eid", "christmas", "baby shower", "newborn", "halloween"],
            "shaadi": ["diwali", "holi", "eid", "christmas", "baby shower", "newborn", "halloween"],
            "diwali": ["valentine", "baby shower", "holi", "eid", "halloween"],
            "valentine": ["diwali", "holi", "eid", "baby shower", "office party"],
            "anniversary": ["baby shower", "diwali", "holi", "eid", "office party"],
            "birthday": ["diwali hamper", "holi hamper", "eid hamper"],
            "office": ["couple", "valentine", "romantic", "baby shower"],
            "corporate": ["couple", "valentine", "romantic", "baby shower"],
            "housewarming": ["valentine", "baby shower", "couple perfume"],
            "baby shower": ["couple perfume", "wine", "choker necklace", "suit set", "diwali hamper"]
        }

        occ_lower = occasion.lower()
        conflicting_terms = []
        for key, terms in conflict_map.items():
            if key in occ_lower:
                conflicting_terms.extend(terms)

        valid_candidates = []
        for item in candidate_products:
            payload = item["payload"]
            title_lower = (payload.get("title") or payload.get("name") or "").lower()
            cat_lower = (payload.get("category_name") or "").lower()
            
            # Check if this item explicitly conflicts with the target occasion
            is_conflicting = any(term in title_lower or term in cat_lower for term in conflicting_terms)
            if not is_conflicting:
                valid_candidates.append(item)

        if not valid_candidates:
            valid_candidates = candidate_products

        # Step 2: External LLM Judge & Rationale Synthesis
        if self.client is not None:
            try:
                candidate_lines = [
                    f"ID: {p['payload'].get('id')} | Title: {p['payload'].get('title')} | Category: {p['payload'].get('category_name')} | Desc: {p['payload'].get('description','')[:100]}"
                    for p in valid_candidates[:16]
                ]
                prompt = (
                    f"You are a strict, high-precision AI e-commerce recommendation validator.\n"
                    f"Occasion: '{occasion}'\n\n"
                    f"Candidate Products:\n" + "\n".join(candidate_lines) + "\n\n"
                    f"Task:\n"
                    f"1. Evaluate each product's genuine suitability for '{occasion}'.\n"
                    f"2. Reject products that do not fit or belong specifically to another occasion.\n"
                    f"3. Select only truly relevant items (maximum {max_items}).\n"
                    f"4. For each selected item, write a concise, professional 1-sentence rationale.\n"
                    f"Rules:\n"
                    f"- DO NOT use any emojis.\n"
                    f"- Respond in valid JSON array format only: [{{\"id\": \"<product_id>\", \"rationale\": \"<1 sentence>\"}}]\n"
                )
                response = self.client.generate_content(prompt, request_options={"timeout": 6.0})
                raw_text = response.text.strip()
                json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                if json_match:
                    filtered_llm = json.loads(json_match.group(0))
                    llm_map = {item['id']: item.get('rationale', '') for item in filtered_llm if 'id' in item}
                    
                    final_products = []
                    for p in valid_candidates:
                        pid = p['payload'].get('id')
                        if pid in llm_map:
                            p['rationale'] = llm_map[pid]
                            final_products.append(p)
                            if len(final_products) >= max_items:
                                break
                    if len(final_products) >= 4:
                        return final_products
            except Exception as e:
                logger.warning(f"[WARNING] LLM filtering notice: {e}. Utilizing deterministic semantic filter.")

        # Step 3: Deterministic fallback selection
        selected = valid_candidates[:max_items]
        for p in selected:
            if "rationale" not in p or not p["rationale"]:
                p["rationale"] = self._synthesize_rule_rationale(occasion, p["payload"])
        return selected

    def generate_rationales(self, occasion: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates clean, human-like recommendation rationales for each product relative to the occasion.
        Ensures strict JSON output and no decorative emoji characters.
        """
        if not products:
            return []

        # If external API is available, generate dynamic synthesized reasoning
        if self.client is not None:
            try:
                product_summaries = [
                    f"ID: {p['payload'].get('id')} | Title: {p['payload'].get('title')} | Category: {p['payload'].get('category_name')}"
                    for p in products
                ]
                prompt = (
                    f"You are an expert product recommendation engine.\n"
                    f"Occasion: '{occasion}'\n"
                    f"Products:\n" + "\n".join(product_summaries) + "\n\n"
                    f"Task: Provide a concise, professional 1-sentence rationale for each product explaining why it fits this occasion.\n"
                    f"Rules:\n"
                    f"- DO NOT use any emojis.\n"
                    f"- Respond in valid JSON format only as a dictionary mapping product ID to rationale string.\n"
                    f"Example format: {{\"id1\": \"Reason here.\", \"id2\": \"Reason here.\"}}"
                )
                response = self.client.generate_content(prompt, request_options={"timeout": 5.0})
                raw_text = response.text.strip()
                # Parse JSON block
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    reasoning_map = json.loads(json_match.group(0))
                    for p in products:
                        pid = p['payload'].get('id')
                        if pid in reasoning_map:
                            p['rationale'] = reasoning_map[pid]
                    return products
            except Exception as e:
                logger.warning(f"[WARNING] Dynamic rationale generation fallback: {e}")

        # Deterministic semantic rationale generation
        for p in products:
            if "rationale" not in p or not p["rationale"]:
                p["rationale"] = self._synthesize_rule_rationale(occasion, p["payload"])

        return products

    def _synthesize_rule_rationale(self, occasion: str, payload: Dict[str, Any]) -> str:
        title = payload.get("title", "")
        category = payload.get("category_name", "Gift")
        brand = payload.get("brand", "")
        desc = payload.get("description", "")

        q = occasion.lower()
        if "wedding" in q or "shaadi" in q or "marriage" in q:
            if "watch" in title.lower():
                return f"Timeless luxury timepiece crafted with refined metallic accents, serving as an enduring wedding keepsake."
            if "suit" in title.lower() or "saree" in title.lower():
                return f"Elegant traditional attire featuring intricate craftsmanship suited for wedding receptions and ceremonies."
            if "jewel" in category.lower() or "choker" in title.lower():
                return f"Statement jewellery designed to elevate festive wedding wardrobes with regal sophistication."
            return f"Curated celebratory gift that complements the grandeur and significance of wedding festivities."

        if "valentine" in q or "anniversary" in q or "romantic" in q or "love" in q:
            if "rose" in title.lower():
                return f"A lasting romantic symbol preserved at peak elegance, offering timeless sentiment for couples."
            if "perfume" in title.lower() or "fragrance" in category.lower():
                return f"Intimate floral and refined fragrance notes crafted to leave a memorable and romantic impression."
            if "chocolate" in title.lower():
                return f"Artisanal confectionery offering rich cocoa indulgence tailored for shared romantic celebrations."
            return f"Thoughtfully curated romantic keepsake that captures warmth, connection, and affection."

        if "diwali" in q or "holi" in q or "eid" in q or "festiv" in q:
            if "dry fruit" in title.lower():
                return f"Traditional dry fruit hamper in festive packaging symbolizing health, prosperity, and celebratory warmth."
            if "candle" in title.lower() or "lamp" in title.lower():
                return f"Aromatic clean-burning illumination bringing warmth and auspicious cheer to festive celebrations."
            return f"Premium festive essential suited for traditional family gifting and celebratory prosperity."

        if "office" in q or "corporate" in q or "work" in q:
            if "pen" in title.lower():
                return f"Sleek executive writing instrument blending modern design with professional daily utility."
            if "shirt" in title.lower() or "wear" in title.lower():
                return f"Tailored professional attire designed for polished comfort throughout the corporate workday."
            if "wellness" in title.lower() or "kit" in title.lower():
                return f"Practical corporate wellness kit supporting work-life balance and employee recognition."
            return f"Refined corporate accessory offering functional elegance suitable for professional gifting."

        # Default fallback rationale
        return f"High quality {category.lower()} item by {brand}, selected for its aesthetic appeal and suitability for {occasion}."
