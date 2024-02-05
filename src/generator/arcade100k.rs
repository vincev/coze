// This is a simplified version of the code at:
//
// https://github.com/openai/tiktoken/blob/main/src/lib.rs
//
// as we need only encoding for inference.
#![allow(clippy::borrow_deref_ref)]

use anyhow::{anyhow, Result};
use fancy_regex::Regex;
use std::collections::HashMap;

fn byte_pair_merge<T>(
    piece: &[u8],
    ranks: &HashMap<Vec<u8>, u32>,
    f: impl Fn(std::ops::Range<usize>) -> T,
) -> Vec<T> {
    // This is a vector of (start, rank).
    // The rank is of the byte pair starting at position start.
    // The rank of the last item in the vector is not a valid value.
    let mut parts: Vec<(usize, u32)> = (0..piece.len() + 1).map(|i| (i, u32::MAX)).collect();

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, u32)>, start_idx: usize, skip: usize| {
            if (start_idx + skip + 2) < parts.len() {
                ranks
                    .get(&piece[parts[start_idx].0..parts[start_idx + skip + 2].0])
                    .copied()
            } else {
                None
            }
        }
    };

    // We look up the ranks once in the beginning and iteratively update
    // them during each merge, which reduces the number of rank lookups.
    for i in 0..parts.len() - 2 {
        match get_rank(&parts, i, 0) {
            Some(rank) => {
                // usize::MAX is a sentinel value and cannot be a valid rank
                debug_assert!(rank != u32::MAX);
                parts[i].1 = rank;
            }
            None => {
                continue;
            }
        };
    }

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // It is important to consider that n is often small (<100), and as such
    // the cache-locality benefits outweigh the algorithmic complexity downsides
    // of the `parts` vector data structure above.

    // Note that we hash bytes, not token pairs. As long as we train BPE the way we
    // currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    loop {
        if parts.len() == 1 {
            break;
        }

        // u32::MAX is a sentinel rank value allowing us to
        // take the min more quickly
        let mut min_rank: (u32, usize) = (u32::MAX, 0);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }

        if min_rank.0 != u32::MAX {
            let i = min_rank.1;

            // NOTE: We are about to remove parts[i + 1]. We do not do it
            // yet because there are cache-locality benefits to updating
            // parts[i] and parts[i-1] before removing, which could thrash
            // the cache. Thus, we update the rank calculation by skipping over
            // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
            parts[i].1 = get_rank(&parts, i, 1).unwrap_or(u32::MAX);
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1, 1).unwrap_or(u32::MAX);
            }

            parts.remove(i + 1);
        } else {
            break;
        }
    }
    let mut out: Vec<T> = Vec::with_capacity(parts.len() - 1);
    for i in 0..parts.len() - 1 {
        out.push(f(parts[i].0..parts[i + 1].0));
    }
    out
}

fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    byte_pair_merge(piece, ranks, |p| ranks[&piece[p.start..p.end]])
}

// Special tokens derived from tokenization_arcade100k.py
// (https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b/tree/main)
fn special_tokens() -> HashMap<String, u32> {
    const SPECIAL_TOKENS: [(&str, u32); 32] = [
        ("<|endoftext|>", 100257),
        ("<|fim_prefix|>", 100258),
        ("<|fim_middle|>", 100259),
        ("<|fim_suffix|>", 100260),
        ("<|fim_pad|>", 100261),
        ("<gh_stars>", 100262),
        ("<filename>", 100263),
        ("<issue_start>", 100264),
        ("<issue_comment>", 100265),
        ("<issue_closed>", 100266),
        ("<jupyter_start>", 100267),
        ("<jupyter_text>", 100268),
        ("<jupyter_code>", 100269),
        ("<jupyter_output>", 100270),
        ("<empty_output>", 100271),
        ("<commit_before>", 100272),
        ("<commit_msg>", 100273),
        ("<commit_after>", 100274),
        ("<reponame>", 100275),
        ("<|endofprompt|>", 100276),
        ("<|im_start|>", 100277),
        ("<|im_end|>", 100278),
        ("<|pause|>", 100279),
        ("<|reg0|>", 100280),
        ("<|reg1|>", 100281),
        ("<|reg2|>", 100282),
        ("<|reg3|>", 100283),
        ("<|reg4|>", 100284),
        ("<|reg5|>", 100285),
        ("<|reg6|>", 100286),
        ("<|reg7|>", 100287),
        ("<|extra0|>", 100288),
    ];

    HashMap::from_iter(SPECIAL_TOKENS.into_iter().map(|(k, v)| (k.to_string(), v)))
}

// Encoder tokens derived from tokenization_arcade100k.py
// (https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b/tree/main)
fn encoder() -> HashMap<Vec<u8>, u32> {
    use base64::prelude::*;

    let tokens = include_str!("arcade100k.tiktoken");
    let mut encoder = HashMap::with_capacity(100_500);

    for line in tokens.lines() {
        let v = line.split(' ').collect::<Vec<_>>();
        assert!(v.len() == 2);
        let key = BASE64_STANDARD.decode(v[0]).unwrap();
        encoder.insert(key, v[1].parse::<u32>().unwrap());
    }

    encoder
}

pub struct Arcade100k {
    encoder: HashMap<Vec<u8>, u32>,
    special_tokens_encoder: HashMap<String, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    special_tokens_decoder: HashMap<u32, Vec<u8>>,
    regex: Regex,
    special_regex: Regex,
}

impl Arcade100k {
    pub fn new() -> Self {
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let regex = Regex::new(pattern).unwrap();

        let encoder = encoder();
        let special_tokens_encoder = special_tokens();

        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|")).unwrap()
        };

        let decoder: HashMap<u32, Vec<u8>> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert!(
            encoder.len() == decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        let special_tokens_decoder: HashMap<u32, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        Arcade100k {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex,
            special_regex,
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut decoded = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes = self
                .decoder
                .get(token)
                .or_else(|| self.special_tokens_decoder.get(token))
                .ok_or_else(|| anyhow!("Unknown token {token}"))?;

            decoded.extend(token_bytes);
        }
        Ok(String::from_utf8_lossy(&decoded).into_owned())
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let special_regex = &self.special_regex;
        let regex = &self.regex;
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                // Find the next allowed special token, if any
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if self
                            .special_tokens_encoder
                            .contains_key(&text[m.start()..m.end()])
                        {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to _encode_ordinary_native
            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                }
                None => break,
            }
        }

        ret
    }

    pub fn get_token(&self, s: &str) -> Option<u32> {
        self.encoder
            .get(s.as_bytes())
            .or_else(|| self.special_tokens_encoder.get(s))
            .copied()
    }
}
