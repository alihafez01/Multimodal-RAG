"""Editable product catalog loading, seeding, and validation."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import Product


class CatalogError(ValueError):
    """Raised when product catalog mappings are unsafe or ambiguous."""


@dataclass(frozen=True)
class CatalogResult:
    path: Path
    products: tuple[Product, ...]
    product_by_document: dict[str, Product]
    generated_products: int


def _stable_product_id(document_id: str) -> str:
    value = f"digital-product-rag:product:{document_id}"
    return "product_" + uuid.uuid5(uuid.NAMESPACE_URL, value).hex[:16]


def _write_catalog(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_product_catalog(
    repository_root: Path,
    documents: Iterable[Any],
    *,
    title_by_document: dict[str, str] | None = None,
) -> CatalogResult:
    """Load the editable catalog, adding one deterministic entry per unmapped PDF."""
    path = repository_root / "config" / "product_catalog.json"
    document_list = list(documents)
    document_ids = {document.document_id for document in document_list}
    if len(document_ids) != len(document_list):
        raise CatalogError("Inventory contains duplicate document IDs.")

    if path.exists():
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError(f"Could not read product catalog {path}: {exc}") from exc
        if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
            raise CatalogError("Product catalog must be an object with schema_version 1.")
        raw_products = catalog.get("products")
        if not isinstance(raw_products, list):
            raise CatalogError("Product catalog 'products' must be a list.")
        raw_products = [dict(item) if isinstance(item, dict) else item for item in raw_products]
        created = 0
    else:
        catalog = {"schema_version": 1, "products": []}
        raw_products = catalog["products"]
        created = 0

    errors: list[str] = []
    product_ids: set[str] = set()
    document_mapping: dict[str, list[int]] = {}
    for index, row in enumerate(raw_products):
        if not isinstance(row, dict):
            errors.append(f"Product entry {index + 1} must be an object.")
            continue
        product_id = row.get("product_id")
        title = row.get("title")
        category = row.get("category", "unknown")
        version = row.get("version")
        mapped_ids = row.get("document_ids")
        if not isinstance(product_id, str) or not product_id.strip():
            errors.append(f"Product entry {index + 1} has no valid product_id.")
        elif product_id in product_ids:
            errors.append(f"Duplicate product_id: {product_id}.")
        else:
            product_ids.add(product_id)
        if not isinstance(title, str) or not title.strip():
            errors.append(f"Product entry {index + 1} has no valid title.")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"Product entry {index + 1} has an invalid category.")
        if version is not None and not isinstance(version, str):
            errors.append(f"Product entry {index + 1} version must be a string or null.")
        if not isinstance(mapped_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in mapped_ids
        ):
            errors.append(f"Product entry {index + 1} document_ids must be a list of IDs.")
            continue
        if len(set(mapped_ids)) != len(mapped_ids):
            errors.append(f"Product {product_id!r} repeats a document ID.")
        for document_id in mapped_ids:
            if document_id not in document_ids:
                errors.append(f"Product {product_id!r} references unknown document {document_id}.")
            document_mapping.setdefault(document_id, []).append(index)

    ambiguous = sorted(key for key, values in document_mapping.items() if len(values) > 1)
    for document_id in ambiguous:
        errors.append(f"Document {document_id} is mapped to multiple products.")
    if errors:
        raise CatalogError("Invalid product catalog:\n- " + "\n- ".join(errors))

    mapped = set(document_mapping)
    titles = title_by_document or {}
    for document in document_list:
        if document.document_id in mapped:
            continue
        title = titles.get(document.document_id) or Path(document.filename).stem
        raw_products.append({
            "product_id": _stable_product_id(document.document_id),
            "title": title,
            "category": "unknown",
            "version": None,
            "document_ids": [document.document_id],
            "metadata_status": "needs_review",
        })
        created += 1
        mapped.add(document.document_id)

    if not path.exists() or created:
        catalog["products"] = raw_products
        _write_catalog(path, catalog)

    products: list[Product] = []
    product_by_document: dict[str, Product] = {}
    for row in raw_products:
        product = Product(
            product_id=row["product_id"],
            title=row["title"],
            category=row.get("category", "unknown"),
            version=row.get("version"),
            document_ids=tuple(row["document_ids"]),
            metadata_status=row.get("metadata_status"),
        )
        products.append(product)
        for document_id in product.document_ids:
            product_by_document[document_id] = product

    unmapped = sorted(document_ids - set(product_by_document))
    if unmapped:
        raise CatalogError("Inventory documents without a product mapping: " + ", ".join(unmapped))
    return CatalogResult(path, tuple(products), product_by_document, created)
