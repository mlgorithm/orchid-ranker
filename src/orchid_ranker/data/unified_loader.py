import os
from typing import Tuple, Dict, Any, Union, Optional

import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class DatasetLoader:
    def __init__(self, encoding: str = "onehot"):
        """
        Generic dataset loader with encoding support.

        Args:
            encoding (str): "onehot" or "label"
        """
        assert encoding in ["onehot", "label"], "Encoding must be 'onehot' or 'label'"
        self.encoding = encoding

        # Back-compat single-encoder fields (used by .load)
        self.encoder: Optional[OneHotEncoder] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Per-entity (users/items) encoders when using YAML
        self.encoders: Dict[str, Any] = {"users": None, "items": None}
        self.label_encoders_map: Dict[str, Dict[str, LabelEncoder]] = {"users": {}, "items": {}}

        # Column groups (for .load and per-entity)
        self.categorical_cols: list[str] = []
        self.numeric_cols: list[str] = []
        self.categorical_cols_map: Dict[str, list[str]] = {"users": [], "items": []}
        self.numeric_cols_map: Dict[str, list[str]] = {"users": [], "items": []}

    # ------------------------ YAML entrypoint ------------------------

    def load_from_yaml(
            self,
            config_path: str,
            dataset: Optional[str] = None,
            encode_side_info: bool = True,
        ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
            """
            Load a dataset bundle (train/val/test + user/item side info) as
            specified in a YAML config.
            """
            cfg = self._read_yaml(config_path)
            ds_name, ds_cfg = self._pick_dataset(cfg, dataset)
            paths = self._resolve_paths(ds_cfg)

            # Load CSVs
            data = {
                "train": pd.read_csv(paths["train"]),
                "val": pd.read_csv(paths["val"]),
                "test": pd.read_csv(paths["test"]),
                "side_information_users": pd.read_csv(paths["side_information_users"]),
                "side_information_items": pd.read_csv(paths["side_information_items"]),
            }

            # Validate interactions
            want_timestamp = bool(ds_cfg.get("interactions", {}).get("timestamp", False))
            for split in ("train", "val", "test"):
                self._validate_interactions(data[split], want_timestamp)

            # Determine (or infer) schemas
            users_cfg = ds_cfg.get("users", {})
            items_cfg = ds_cfg.get("items", {})

            u_cat, u_num = list(users_cfg.get("categorical", [])), list(users_cfg.get("numeric", []))
            i_cat, i_num = list(items_cfg.get("categorical", [])), list(items_cfg.get("numeric", []))

            if not u_cat and not u_num:
                u_cat, u_num = self._infer_schema(data["side_information_users"])
            if not i_cat and not i_num:
                i_cat, i_num = self._infer_schema(data["side_information_items"])

            # Store per-entity columns
            self.categorical_cols_map["users"], self.numeric_cols_map["users"] = u_cat, u_num
            self.categorical_cols_map["items"], self.numeric_cols_map["items"] = i_cat, i_num

            # Optionally encode side info (separately for users and items)
            if encode_side_info:
                users_proc, u_meta = self._encode_entity(
                    entity="users",
                    df=data["side_information_users"],
                    categorical_cols=u_cat,
                    numeric_cols=u_num,
                )
                items_proc, i_meta = self._encode_entity(
                    entity="items",
                    df=data["side_information_items"],
                    categorical_cols=i_cat,
                    numeric_cols=i_num,
                )
                data_proc = dict(data)
                data_proc["side_information_users"] = users_proc
                data_proc["side_information_items"] = items_proc
            else:
                data_proc = data
                u_meta = {"encoding": None}
                i_meta = {"encoding": None}

            # --------- Build meta ---------
            meta = {
                "dataset": ds_name,
                "paths": paths,
                "schemas": {
                    "users": {"categorical": u_cat, "numeric": u_num},
                    "items": {"categorical": i_cat, "numeric": i_num},
                },
                "encoders": {
                    "users": self.encoders["users"],
                    "items": self.encoders["items"],
                    "users_label": self.label_encoders_map["users"],
                    "items_label": self.label_encoders_map["items"],
                },
                "users_meta": u_meta,
                "items_meta": i_meta,
                "config": ds_cfg,   # <-- NEW: keep raw dataset config for agents/privacy
            }

            return data_proc, meta


    # ------------------------ Original single-DF API ------------------------

    def load(self, data: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load dataset and encode categorical features (single table).
        Kept for backward compatibility.
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("data must be a CSV path or a Pandas DataFrame")

        # Detect categorical vs numeric
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_cols = df.select_dtypes(exclude=["object", "category"]).columns.tolist()

        if self.encoding == "onehot":
            df_processed = self._encode_onehot(df)
        else:
            df_processed = self._encode_label(df)

        metadata = {
            "encoding": self.encoding,
            "encoder": self.encoder,
            "label_encoders": self.label_encoders,
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "all_cols": df_processed.columns.tolist()
        }
        return df_processed, metadata

    # ------------------------ Public helpers ------------------------

    def transform_row(self, entity: str, row: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single side-info row (dict) into a numeric vector
        using the fitted encoder for the given entity ("users" or "items").
        """
        entity = entity.lower()
        if self.encoding == "onehot":
            enc: OneHotEncoder = self.encoders[entity]
            cats = self.categorical_cols_map[entity]
            nums = self.numeric_cols_map[entity]
            # Numeric part (+ keep order)
            num_vals = [self._to_float(row.get(c, np.nan)) for c in nums]
            # Categorical part
            if cats:
                enc_in = pd.DataFrame([{c: str(row.get(c, "")) for c in cats}])
                oh = enc.transform(enc_in).astype(float)
                vec = np.concatenate([np.array(num_vals, dtype=float), oh.ravel()], axis=0)
            else:
                vec = np.array(num_vals, dtype=float)
            return vec
        else:
            # Label-encoded: concatenate numeric + label cols (as floats)
            le_map = self.label_encoders_map[entity]
            cats = self.categorical_cols_map[entity]
            nums = self.numeric_cols_map[entity]
            out = [self._to_float(row.get(c, np.nan)) for c in nums]
            for c in cats:
                le = le_map[c]
                val = str(row.get(c, ""))
                try:
                    out.append(float(le.transform([val])[0]))
                except Exception:
                    out.append(np.nan)
            return np.array(out, dtype=float)

    def decode_row(self, row: Dict[str, Any], entity: str = "users") -> Dict[str, Any]:
        """
        Decode a sanitized/encoded row back into human-readable categories.
        Works with both One-Hot and Label encoding.

        Args:
            row: mapping of feature_name -> value (e.g., a model-ready dict)
            entity: "users" or "items" (selects the right encoder set)
        """
        entity = entity.lower()
        decoded: Dict[str, Any] = {}

        # Numeric features: echo through if present
        for col in self.numeric_cols_map.get(entity, []):
            decoded[col] = row.get(col, None)

        if self.encoding == "onehot" and self.encoders.get(entity) is not None:
            enc: OneHotEncoder = self.encoders[entity]
            cats = self.categorical_cols_map[entity]
            feature_names = enc.get_feature_names_out(cats)
            # For each categorical, find its one-hot group and pick the suffix with highest value
            for cat in cats:
                group_cols = [fn for fn in feature_names if fn.startswith(cat + "_")]
                if not group_cols:
                    continue
                vals = np.array([row.get(col, 0) for col in group_cols], dtype=float)
                idx = int(vals.argmax()) if len(vals) else 0
                # category is the suffix after "cat_"
                decoded[cat] = group_cols[idx][len(cat) + 1 :]

        elif self.encoding == "label":
            for col, le in self.label_encoders_map.get(entity, {}).items():
                val = row.get(col, None)
                if val is not None and isinstance(val, (int, float)):
                    try:
                        decoded[col] = le.inverse_transform([int(round(val))])[0]
                    except Exception:
                        decoded[col] = "[UNKNOWN]"
        
        return decoded

    # ------------------------ Internal helpers ------------------------

    def _encode_entity(
        self,
        entity: str,
        df: pd.DataFrame,
        categorical_cols: list[str],
        numeric_cols: list[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit & transform a side-info table for a given entity (users/items).
        Returns processed DataFrame and a small meta dict.
        """
        entity = entity.lower()
        self.categorical_cols_map[entity] = list(categorical_cols)
        self.numeric_cols_map[entity] = list(numeric_cols)

        if self.encoding == "onehot":
            enc = self._make_ohe()
            if categorical_cols:
                X_cat = enc.fit_transform(df[categorical_cols].astype(str))
                cat_names = enc.get_feature_names_out(categorical_cols)
                df_cat = pd.DataFrame(X_cat, columns=cat_names, index=df.index)
                df_num = df[numeric_cols] if numeric_cols else pd.DataFrame(index=df.index)
                df_out = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
            else:
                df_out = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
            self.encoders[entity] = enc
            meta = {
                "encoding": "onehot",
                "feature_names": df_out.columns.tolist(),
                "categorical_cols": categorical_cols,
                "numeric_cols": numeric_cols,
            }

        else:  # label
            df_out = df.copy()
            le_map: Dict[str, LabelEncoder] = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_out[col] = le.fit_transform(df[col].astype(str))
                le_map[col] = le
            # keep only numeric + (now) numeric-encoded categoricals
            keep = list(numeric_cols) + list(categorical_cols)
            df_out = df_out[keep] if keep else pd.DataFrame(index=df.index)
            self.label_encoders_map[entity] = le_map
            meta = {
                "encoding": "label",
                "feature_names": df_out.columns.tolist(),
                "categorical_cols": categorical_cols,
                "numeric_cols": numeric_cols,
            }
        
        id_candidates = ["u", "i", "user_id", "item_id", "id"]
        id_col = next((c for c in id_candidates if c in df.columns), None)
        if id_col is not None:
            # Keep as first column; cast to int if possible
            try:
                ids = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
            except Exception:
                ids = df[id_col]
            df_out.insert(0, id_col, ids)
        return df_out, meta

    def _encode_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply One-Hot Encoding to categorical columns (single-table path)."""
        if self.categorical_cols:
            self.encoder = self._make_ohe()
            encoded = self.encoder.fit_transform(df[self.categorical_cols].astype(str))
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.categorical_cols),
                index=df.index,
            )
            df_num = df[self.numeric_cols] if self.numeric_cols else pd.DataFrame(index=df.index)
            df_processed = pd.concat([df_num.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        else:
            df_processed = df.copy()
        return df_processed

    def _encode_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Label Encoding to categorical columns (single-table path)."""
        df_processed = df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df_processed

    @staticmethod
    def _read_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _pick_dataset(cfg: Dict[str, Any], override: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        name = override or cfg.get("run", {}).get("dataset")
        if not name:
            raise ValueError("No dataset specified (use --dataset or run.dataset in YAML)")
        ds_cfg = cfg.get("datasets", {}).get(name)
        if not ds_cfg:
            raise ValueError(f"Dataset '{name}' missing in YAML under datasets.")
        return name, ds_cfg

    @staticmethod
    def _resolve_paths(ds_cfg: Dict[str, Any]) -> Dict[str, str]:
        p = ds_cfg.get("paths", {})
        base = p.get("base_dir", ".")
        required = ["train", "val", "test", "side_information_users", "side_information_items"]
        out = {}
        for k in required:
            rel = p.get(k)
            if rel is None:
                raise KeyError(f"paths.{k} missing in YAML dataset block")
            out[k] = rel if os.path.isabs(rel) else os.path.join(base, rel)
        return out

    @staticmethod
    def _validate_interactions(df: pd.DataFrame, want_timestamp: bool) -> None:
        needed = ["u", "i", "label"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Interactions file missing columns: {missing}")
        if want_timestamp and "timestamp" not in df.columns:
            raise ValueError("Configured to use timestamp but 'timestamp' column not found.")

    @staticmethod
    def _infer_schema(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
        cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num = df.select_dtypes(exclude=["object", "category"]).columns.tolist()
        for rid in ("u", "i", "id"):
            if rid in cat:
                cat.remove(rid)
            if rid in num:
                num.remove(rid)
        return cat, num

    @staticmethod
    def _to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return np.nan

    @staticmethod
    def _make_ohe() -> OneHotEncoder:
        # sklearn >=1.2 uses 'sparse_output', older uses 'sparse'
        try:
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            return OneHotEncoder(sparse=False, handle_unknown="ignore")




def load_dataset(dataset: str, base_path: str, config_path: str, *, encoding: str = "onehot", encode_side_info: bool = True):
    """Convenience wrapper that returns the processed dataset bundle.

    Args:
        dataset: Dataset key to load (must exist in the YAML file).
        base_path: Optional override for the dataset's ``paths.base_dir``; pass
            ``None`` to use the value from the YAML file.
        config_path: Path to the YAML configuration.
        encoding: ``"onehot"`` (default) or ``"label"``.
        encode_side_info: Whether to encode side-information tables.

    Returns:
        Dictionary containing train/val/test splits, processed side-information
        tables, and the raw dataset configuration.
    """
    loader = DatasetLoader(encoding=encoding)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ds_name = dataset or cfg.get('run', {}).get('dataset')
    if not ds_name:
        raise ValueError("Dataset name must be provided or configured under run.dataset")

    if base_path:
        cfg.setdefault('datasets', {}).setdefault(ds_name, {}).setdefault('paths', {})['base_dir'] = base_path

    import tempfile, os

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tmp:
        tmp_path = tmp.name
        yaml.safe_dump(cfg, tmp)

    try:
        data, meta = loader.load_from_yaml(tmp_path, dataset=ds_name, encode_side_info=encode_side_info)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return {
        'train': data['train'],
        'val': data['val'],
        'test': data['test'],
        'side_info_user': data['side_information_users'],
        'side_info_item': data['side_information_items'],
        'meta': meta,
        'config': meta.get('config', {}),
    }
