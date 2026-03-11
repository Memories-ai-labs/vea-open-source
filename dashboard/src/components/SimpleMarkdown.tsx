import React from 'react';

/**
 * Lightweight markdown renderer for scratchpad content.
 * Handles: ## headings, **bold**, - lists, and paragraphs.
 * No external dependencies.
 */
export function SimpleMarkdown({ text }: { text: string }) {
  const blocks = parseBlocks(text);
  return <div className="md-root">{blocks.map((b, i) => renderBlock(b, i))}</div>;
}

type Block =
  | { type: 'heading'; level: number; text: string }
  | { type: 'list'; items: string[] }
  | { type: 'paragraph'; text: string };

function parseBlocks(raw: string): Block[] {
  const lines = raw.split('\n');
  const blocks: Block[] = [];
  let listBuf: string[] = [];

  function flushList() {
    if (listBuf.length > 0) {
      blocks.push({ type: 'list', items: [...listBuf] });
      listBuf = [];
    }
  }

  for (const line of lines) {
    const trimmed = line.trimStart();

    // Heading
    const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      flushList();
      blocks.push({ type: 'heading', level: hMatch[1].length, text: hMatch[2] });
      continue;
    }

    // List item (- or * prefixed, or indented sub-items like "  - ")
    const liMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (liMatch) {
      listBuf.push(liMatch[1]);
      continue;
    }

    // Blank line
    if (trimmed === '') {
      flushList();
      continue;
    }

    // Regular paragraph text
    flushList();
    // Merge with previous paragraph if it exists
    const prev = blocks[blocks.length - 1];
    if (prev && prev.type === 'paragraph') {
      prev.text += ' ' + trimmed;
    } else {
      blocks.push({ type: 'paragraph', text: trimmed });
    }
  }
  flushList();
  return blocks;
}

function renderInline(text: string): React.ReactNode[] {
  // Split on **bold** markers
  const parts: React.ReactNode[] = [];
  const regex = /\*\*(.+?)\*\*/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <strong key={match.index} style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
        {match[1]}
      </strong>
    );
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

function renderBlock(block: Block, key: number): React.ReactNode {
  if (block.type === 'heading') {
    const sizes: Record<number, string> = { 1: '15px', 2: '14px', 3: '12px', 4: '11px' };
    return (
      <div
        key={key}
        style={{
          fontSize: sizes[block.level] || '12px',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginTop: key === 0 ? 0 : block.level <= 2 ? '18px' : '12px',
          marginBottom: '6px',
          letterSpacing: '-0.01em',
          fontFamily: 'var(--font-sans)',
        }}
      >
        {renderInline(block.text)}
      </div>
    );
  }

  if (block.type === 'list') {
    return (
      <ul
        key={key}
        style={{
          margin: '4px 0',
          paddingLeft: '16px',
          listStyle: 'none',
        }}
      >
        {block.items.map((item, i) => (
          <li
            key={i}
            style={{
              fontSize: '11px',
              lineHeight: 1.7,
              color: 'var(--text-secondary)',
              position: 'relative',
              paddingLeft: '10px',
            }}
          >
            <span
              style={{
                position: 'absolute',
                left: 0,
                color: 'var(--text-muted)',
                fontSize: '7px',
                top: '6px',
              }}
            >
              ●
            </span>
            {renderInline(item)}
          </li>
        ))}
      </ul>
    );
  }

  // paragraph
  return (
    <p
      key={key}
      style={{
        fontSize: '11px',
        lineHeight: 1.7,
        color: 'var(--text-secondary)',
        margin: '6px 0',
      }}
    >
      {renderInline(block.text)}
    </p>
  );
}
