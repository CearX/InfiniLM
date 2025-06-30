use llama_cu::{Cache, CacheParts, SampleArgs, Session, SessionId, Terminal, utok};
use std::{
    collections::BTreeMap,
    iter::zip,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
    time::Instant,
};

#[derive(Default)]
#[repr(transparent)]
pub(super) struct CacheManager(BTreeMap<Instant, (Vec<utok>, Cache<CacheParts>)>);

impl CacheManager {
    pub fn send(
        &mut self,
        terminal: &Terminal,
        tokens: Vec<utok>,
        sample_args: SampleArgs,
        max_tokens: usize,
    ) -> (SessionId, Vec<utok>) {
        static SESSION_ID: AtomicUsize = AtomicUsize::new(0);
        let id = SessionId(SESSION_ID.fetch_add(1, SeqCst));

        let use_cache = &tokens[..tokens.len() - 1];
        let best_cache = self
            .0
            .iter()
            .map(|(key, (history, _))| (*key, common_len(history, use_cache)))
            .max_by_key(|&(_, len)| len);

        let cache = match best_cache {
            Some((key, pos)) => {
                let (_, mut cache) = self.0.remove(&key).unwrap();
                cache.len = pos;
                cache
            }
            None => terminal.new_cache(),
        };
        let pos = cache.len;
        terminal.start(
            Session {
                id,
                sample_args,
                cache,
            },
            &tokens[pos..],
            max_tokens,
            None,
        );
        (id, tokens)
    }

    pub fn insert(&mut self, tokens: Vec<utok>, cache: Cache<CacheParts>) {
        assert!(self.0.insert(Instant::now(), (tokens, cache)).is_none())
    }
}

fn common_len<T: Eq>(a: &[T], b: &[T]) -> usize {
    zip(a, b).take_while(|(a, b)| a == b).count()
}
