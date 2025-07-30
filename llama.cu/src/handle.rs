use crate::op::ModuleKey;
use cublas::Cublas;
use cuda::{CurrentCtx, Module, Rtc};
use std::collections::HashMap;

#[cfg(nccl)]
use nccl::Communicator;

pub(crate) struct Handle<'ctx> {
    pub ctx: &'ctx CurrentCtx,
    pub cublas: Cublas<'ctx>,
    pub modules: HashMap<Box<[ModuleKey]>, Module<'ctx>>,
    #[cfg(nccl)]
    pub comm: Option<Communicator>,
}

impl<'ctx> Handle<'ctx> {
    pub fn new(ctx: &'ctx CurrentCtx) -> Self {
        Self {
            ctx,
            cublas: Cublas::new(ctx),
            modules: HashMap::new(),
            #[cfg(nccl)]
            comm: None,
        }
    }

    #[cfg(nccl)]
    pub fn with_comm(ctx: &'ctx CurrentCtx, comm: Communicator) -> Self {
        Self {
            ctx,
            cublas: Cublas::new(ctx),
            modules: HashMap::new(),
            comm: Some(comm),
        }
    }

    pub fn compile(&mut self, key: Box<[ModuleKey]>, code: impl FnOnce() -> String) -> &Module {
        self.modules.entry(key).or_insert_with(|| {
            let program = Rtc::new()
                .arch(self.ctx.dev().compute_capability())
                .compile(&code())
                .unwrap();
            self.ctx.load(&program)
        })
    }

    pub fn rank(&self) -> usize {
        #[cfg(nccl)]
        {
            self.comm.as_ref().map_or(0, Communicator::rank)
        }
        #[cfg(not(nccl))]
        {
            0
        }
    }
}
