use indexed_db_futures::prelude::*;
use js_sys::Uint8Array;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum RatchetDBError {
    #[error("DomException {name} ({code}): {message}")]
    DomException {
        name: String,
        message: String,
        code: u16,
    },
    #[error(transparent)]
    SerdeError(#[from] serde_wasm_bindgen::Error),
}

impl From<indexed_db_futures::web_sys::DomException> for RatchetDBError {
    fn from(e: indexed_db_futures::web_sys::DomException) -> Self {
        Self::DomException {
            name: e.name(),
            message: e.message(),
            code: e.code(),
        }
    }
}

pub struct RatchetDB {
    pub(crate) inner: IdbDatabase,
}

type Result<A, E = RatchetDBError> = std::result::Result<A, E>;

impl RatchetDB {
    pub const DB_VERSION: u32 = 1;
    pub const DB_NAME: &'static str = "ratchet";
    pub const MODEL_STORE: &'static str = "models";
    pub const TOKENIZER_STORE: &'static str = "tokenizers";

    fn serialize(o: &impl Serialize) -> Result<JsValue> {
        serde_wasm_bindgen::to_value(o).map_err(|e| e.into())
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(o: Option<JsValue>) -> Result<Option<T>> {
        o.map(serde_wasm_bindgen::from_value)
            .transpose()
            .map_err(|e| e.into())
    }

    pub async fn open() -> Result<Self, RatchetDBError> {
        let mut db_req: OpenDbRequest = IdbDatabase::open_u32(Self::DB_NAME, Self::DB_VERSION)?;

        db_req.set_on_upgrade_needed(Some(|evt: &IdbVersionChangeEvent| -> Result<(), JsValue> {
            let create_if_needed =
                |evt: &IdbVersionChangeEvent, store_key: &'static str| -> Result<(), JsValue> {
                    if let None = evt.db().object_store_names().find(|n| n == store_key) {
                        evt.db().create_object_store(store_key)?;
                    }
                    Ok(())
                };
            create_if_needed(evt, Self::MODEL_STORE)?;
            create_if_needed(evt, Self::TOKENIZER_STORE)?;
            Ok(())
        }));

        Ok(Self {
            inner: db_req.await?,
        })
    }

    pub async fn get_model(&self, key: &ModelKey) -> Result<Option<StoredModel>> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::MODEL_STORE, IdbTransactionMode::Readonly)?;
        let store = tx.object_store(Self::MODEL_STORE)?;
        let serial_key = Self::serialize(key)?;
        let req = store.get(&serial_key)?.await?;
        Self::deserialize(req)
    }

    pub async fn put_model(&self, key: &ModelKey, model: StoredModel) -> Result<()> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::MODEL_STORE, IdbTransactionMode::Readwrite)?;
        let store = tx.object_store(Self::MODEL_STORE)?;
        store
            .put_key_val(&Self::serialize(key)?, &Self::serialize(&model)?)?
            .await?;
        Ok(())
    }

    pub async fn get_tokenizer<S: AsRef<str>>(
        &self,
        repo_id: S,
    ) -> Result<Option<StoredTokenizer>> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::TOKENIZER_STORE, IdbTransactionMode::Readonly)?;
        let store = tx.object_store(Self::TOKENIZER_STORE)?;
        let req = store.get(&Self::serialize(&repo_id.as_ref())?)?.await?;
        Self::deserialize(req)
    }

    pub async fn put_tokenizer<S: AsRef<str>>(
        &self,
        repo_id: S,
        tokenizer: StoredTokenizer,
    ) -> Result<()> {
        let tx = self
            .inner
            .transaction_on_one_with_mode(Self::TOKENIZER_STORE, IdbTransactionMode::Readwrite)?;
        let store = tx.object_store(Self::TOKENIZER_STORE)?;
        store
            .put_key_val(
                &Self::serialize(&repo_id.as_ref())?,
                &Self::serialize(&tokenizer)?,
            )?
            .await?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ModelKey {
    pub repo_id: String,
    pub model_id: String,
}

impl serde::Serialize for ModelKey {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let s = format!("{}:{}", self.repo_id, self.model_id);
        serializer.serialize_str(&s)
    }
}

impl<'de> serde::Deserialize<'de> for ModelKey {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let mut parts = s.split(':');
        let repo_id = parts.next().unwrap().to_string();
        let model_id = parts.next().unwrap().to_string();
        Ok(Self { repo_id, model_id })
    }
}

impl ModelKey {
    pub fn new<S: ToString>(repo_id: S, model_id: S) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            model_id: model_id.to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredModel {
    pub repo_id: String,
    pub model_id: String,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub bytes: Uint8Array,
}

impl StoredModel {
    pub fn new(key: &ModelKey, bytes: Uint8Array) -> Self {
        Self {
            repo_id: key.repo_id.clone(),
            model_id: key.model_id.clone(),
            bytes,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredTokenizer {
    pub repo_id: String,
    pub tokenizer_id: String,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub bytes: Uint8Array,
}
