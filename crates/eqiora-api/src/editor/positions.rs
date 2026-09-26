use super::EditorPosition;

fn starts(source: &str) -> impl Iterator<Item = usize> + '_ {
    let bytes = source.as_bytes();
    let mut offset = 0;
    std::iter::once(0).chain(std::iter::from_fn(move || {
        while offset < bytes.len() {
            let width = match bytes[offset] {
                b'\r' if bytes.get(offset + 1) == Some(&b'\n') => 2,
                b'\r' | b'\n' => 1,
                _ => {
                    offset += 1;
                    continue;
                }
            };
            offset += width;
            return Some(offset);
        }
        None
    }))
}

pub(super) fn line_starts(source: &str) -> Vec<u32> {
    starts(source)
        .filter_map(|start| u32::try_from(start).ok())
        .collect()
}

pub(super) fn line_end(source: &str, start: usize, mut end: usize) -> usize {
    if end > start && source.as_bytes().get(end - 1) == Some(&b'\n') {
        end -= 1;
    }
    if end > start && source.as_bytes().get(end - 1) == Some(&b'\r') {
        end -= 1;
    }
    end
}

pub(super) fn utf16_offset(
    source: &str,
    start: usize,
    end: usize,
    character: u32,
    clamp: bool,
) -> Option<usize> {
    let target = usize::try_from(character).ok()?;
    let mut utf16 = 0_usize;
    for (relative, character) in source[start..end].char_indices() {
        if utf16 == target {
            return Some(start + relative);
        }
        utf16 = utf16.checked_add(character.len_utf16())?;
        if utf16 > target {
            return None;
        }
    }
    (clamp || utf16 == target).then_some(end)
}

pub(super) fn edit_offset(source: &str, position: EditorPosition) -> Option<usize> {
    let mut lines = starts(source);
    let start = lines.nth(usize::try_from(position.line()).ok()?)?;
    let end = line_end(source, start, lines.next().unwrap_or(source.len()));
    utf16_offset(source, start, end, position.character(), true)
}
