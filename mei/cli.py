"""CLI interface for Environmental & Financial Framework (and legacy MEI compatibility)."""

import argparse
import sys
from pathlib import Path
from src.preprocessing.codec import EnvFinCodec


def main():
    parser = argparse.ArgumentParser(
        description="Environmental & Financial Intelligence CLI (MEI Compatibility)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    setup_parser = subparsers.add_parser(
        "setup", help="Validate YAML schemas, mapping configurations, and item codecs"
    )
    setup_parser.add_argument(
        "--schema", default="config/schema_envfin.yaml", help="Path to schema YAML"
    )
    setup_parser.add_argument(
        "--mapping", default="config/mapping_envfin.yaml", help="Path to mapping YAML"
    )

    args = parser.parse_args()

    if args.command == "setup":
        try:
            schema_path = Path(args.schema)
            if not schema_path.exists():
                schema_path = Path("configs/schema_envfin.yaml")
            mapping_path = Path(args.mapping)
            if not mapping_path.exists():
                mapping_path = Path("configs/mapping_envfin.yaml")

            codec = EnvFinCodec(schema_path=schema_path, mapping_path=mapping_path)
            n_items = len(codec.items)
            n_indices = len(codec.derived_indices)
            n_outcomes = len(codec.outcomes)

            assert n_items == 17, f"Expected 17 canonical survey items, found {n_items}"
            assert n_indices == 4, f"Expected 4 derived indices, found {n_indices}"
            assert n_outcomes >= 2, f"Expected at least 2 outcomes, found {n_outcomes}"

            # Verify every item has a pressure direction and can encode/decode all levels
            for name, meta in codec.items.items():
                assert "pressure_direction" in meta, f"Item '{name}' missing pressure_direction"
                assert meta.get("role") == "feature", f"Item '{name}' role must be 'feature'"
                for lvl in meta.get("levels", []):
                    code = lvl["code"]
                    label = lvl["label"]
                    enc = codec.encode(name, label)
                    dec = codec.decode(name, enc)
                    assert dec == label, f"Codec mismatch for '{name}': {label} -> {enc} -> {dec}"

            print(
                f"[SUCCESS] Environmental & Financial Schema Validated: "
                f"{n_items} items, {n_indices} derived indices, {n_outcomes} outcomes. All codecs verified."
            )
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
