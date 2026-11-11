import json
import re
from pathlib import Path
from typing import List, Dict, Any
from backend.app.schemas import ProductPayload

# Category slug / id to human readable name map
CATEGORY_MAP = {
    "cml0ug1d6003to201xuwjkmcn": "Gift Packaging & Boxes",
    "cmkz57wza0012ns0162oh4obo": "Gourmet Chocolates & Treats",
    "cmkz54pbm000vns019mokm8dp": "Fresh Flowers & Bouquets",
    "cml256swj00elma01jdp2zgfb": "Festive Gift Hampers",
    "cml0g5j3g0002sc01qobq1n72": "Corporate & Wellness Gifts",
    "cml0uasxm003eo201thwvbg5l": "Fragrance & Aromatherapy",
    "cmkz592k20018ns012ju457bg": "Watches & Timepieces",
    "cml0g6qev0007sc01d1wj96mo": "Fashion & Ethnic Apparel",
    "cml0olc910039pw01177zda8g": "Jewellery & Accessories",
    "cml0r659g008tpw018zqjkdkb": "Sustainable & Eco Decor",
    "cml0qw58i008jpw01p5a85pr7": "Eco-Friendly Stationery & Crafts",
    "cml0r4bxr008opw01aktfdvmx": "Organic Wellness & Self-Care",
    "cml0u88h80038o201hhcw0q2k": "Creative DIY & Craft Kits",
    "cml0u5vji0032o201xpg166lu": "Plush Toys & Soft Gifts",
    "cml0tctfc002xo201hpycr57n": "Musical Toys & Rhythm Sets",
    "cml1xsl8p000ald01fjlwo2lo": "Romantic Gifts & Keepsakes",
    "cml1xxy3k000gld01lmldeic1": "Personalized Keepsakes",
    "cml1y3ir8000hma01zboy5gts": "Couple Gifts & Connect Items",
    "cml0ue769003no201f7vi986z": "Gift Cards & Vouchers",
    "cml258vol00eqma01a56hvc3y": "Milestone & Celebration Gifts",
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_catalog(file_path: Path) -> List[ProductPayload]:
    """
    Loads and normalizes product payloads following the take-home assignment specification.
    Supports both payload formats (standard assignment schema and provided JSON fields).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Catalog file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    products: List[ProductPayload] = []

    for item in raw_data:
        # Standard extraction pattern as described in Assignment Section 2
        product_id = str(item.get("product_id") or item.get("productId") or item.get("id") or "")
        vendor_id = str(item.get("vendor_id") or item.get("vendorId") or item.get("shopId") or "")
        vendor_category = str(item.get("vendor_category") or item.get("categoryId") or "")
        vendor_category_desc = item.get("vendor_category_desc")

        title = clean_text(item.get("title") or item.get("name") or "")
        description = clean_text(item.get("description") or "")
        short_desc = clean_text(item.get("shortDescription") or "")

        files_raw = item.get("files") or []
        main_image = item.get("mainImage")
        if main_image and main_image not in files_raw:
            files_raw.insert(0, main_image)

        # Fallback image if files empty
        if not files_raw and main_image:
            files_raw = [main_image]

        brand = clean_text(item.get("brand") or "Curated Collection")

        # Category resolution
        category_name = vendor_category_desc or CATEGORY_MAP.get(vendor_category) or "Lifestyle & Gifts"

        # Price parsing
        raw_price = item.get("price", "0")
        try:
            price_val = float(raw_price)
        except (ValueError, TypeError):
            price_val = 0.0

        raw_discount = item.get("discountPercentage", "0")
        try:
            discount_val = float(raw_discount)
        except (ValueError, TypeError):
            discount_val = 0.0

        stock_val = int(item.get("stock", 50))
        status = item.get("status", "ACTIVE")

        # Build comprehensive searchable context for dense vector generation
        searchable_text = (
            f"Product: {title}. "
            f"Category: {category_name}. "
            f"Brand: {brand}. "
            f"Description: {description}. "
            f"Summary: {short_desc}."
        )

        product = ProductPayload(
            id=product_id,
            shop_id=vendor_id,
            category_id=vendor_category,
            category_name=category_name,
            title=title,
            description=description,
            short_description=short_desc,
            main_image=files_raw[0] if files_raw else None,
            files=files_raw,
            brand=brand,
            sku=item.get("sku"),
            price=price_val,
            discount_percentage=discount_val,
            stock=stock_val,
            status=status,
            searchable_text=searchable_text
        )
        products.append(product)

    return products
