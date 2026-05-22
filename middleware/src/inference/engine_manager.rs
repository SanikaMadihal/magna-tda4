use std::sync::Arc;

use crate::backends::cpu::adapter::CpuAdapter;
use crate::backends::ti::adapter::TiAdapter;
use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer};
use crate::utils::errors::MiddlewareResult;

pub struct EngineManager {
    backend: Arc<Box<dyn InferenceBackend>>,
}

impl EngineManager {
    pub fn new(backend_type: &str) -> MiddlewareResult<Self> {
        let backend: Box<dyn InferenceBackend> = match backend_type {
            #[cfg(feature = "ti")]
            "ti" | _ => {
                Box::new(TiAdapter::new())
            }

            #[cfg(not(feature = "ti"))]
            _ => {
                Box::new(CpuAdapter::new())
            }
        };

        Ok(Self {
            backend: Arc::new(backend),
        })
    }

    pub fn load_engine(&mut self, path: &str) -> MiddlewareResult<EngineInfo> {
        let info = self.backend.load_engine(path)?;
        self.backend.allocate_buffers()?;
        Ok(info)
    }

    pub fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        self.backend.infer(inputs)
    }

    pub fn engine_info(&self) -> Option<&EngineInfo> {
        // EngineInfo is behind Arc<Box<dyn>>, clone and return ref via leaked — use owned instead
        self.backend.engine_info().map(|_| {
            // We store engine_info inside the adapter, so we delegate
            // public_api.rs calls .cloned() on this already
            unreachable!("use engine_info_owned")
        })
    }

    pub fn engine_info_owned(&self) -> Option<EngineInfo> {
        self.backend.engine_info()
    }

    pub fn backend_name(&self) -> &str {
        self.backend.backend_name()
    }

    pub fn release(&mut self) -> MiddlewareResult<()> {
        self.backend.release()
    }

    pub fn get_backend(&self) -> Arc<Box<dyn InferenceBackend>> {
        Arc::clone(&self.backend)
    }
}
