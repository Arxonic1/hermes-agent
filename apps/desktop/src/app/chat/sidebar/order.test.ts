import { describe, expect, it } from 'vitest'

import { findActiveItem, orderByIds, reconcileOrderIds, resolveManualSessionOrderIds, sameIds } from './order'

describe('findActiveItem', () => {
  const items = [
    { id: 'newest', lineageRoot: null },
    { id: 'current-tip', lineageRoot: 'current-root' },
    { id: 'oldest', lineageRoot: null }
  ]

  const ids = (item: (typeof items)[number]) => [item.id, item.lineageRoot]

  it('selects the active item independently of its position in a long list', () => {
    expect(findActiveItem(items, ids, 'current-tip')).toBe(items[1])
  })

  it('selects a compressed continuation by its focused lineage-root id', () => {
    expect(findActiveItem(items, ids, 'current-root')).toBe(items[1])
  })

  it('returns undefined when there is no active match', () => {
    expect(findActiveItem(items, ids, 'missing')).toBeUndefined()
    expect(findActiveItem(items, ids, null)).toBeUndefined()
  })

  it('returns the first match in iteration order when several items share the active id', () => {
    const tie = [
      { id: 'root', lineageRoot: null },
      { id: 'tip', lineageRoot: 'root' }
    ]

    expect(findActiveItem(tie, item => [item.id, item.lineageRoot], 'root')).toBe(tie[0])
  })
})

describe('resolveManualSessionOrderIds', () => {
  it('clears legacy auto-seeded order until the user manually reorders sessions', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older'], ['older', 'newest'], false)).toEqual([])
  })

  it('keeps a manual order and surfaces newly seen sessions first', () => {
    expect(resolveManualSessionOrderIds(['newest', 'older', 'oldest'], ['oldest', 'older'], true)).toEqual([
      'newest',
      'oldest',
      'older'
    ])
  })

  it('clears manual order when none of the saved ids still exist', () => {
    expect(resolveManualSessionOrderIds(['newest'], ['gone'], true)).toEqual([])
  })
})

describe('orderByIds', () => {
  const id = (item: { id: string }) => item.id

  it('returns items untouched when no order is given', () => {
    const items = [{ id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, [])).toBe(items)
  })

  it('reorders by the given ids and drops missing ones', () => {
    const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    expect(orderByIds(items, id, ['c', 'gone', 'a'])).toEqual([{ id: 'b' }, { id: 'c' }, { id: 'a' }])
  })

  it('surfaces items absent from the order first', () => {
    const items = [{ id: 'fresh' }, { id: 'a' }, { id: 'b' }]
    expect(orderByIds(items, id, ['b', 'a'])).toEqual([{ id: 'fresh' }, { id: 'b' }, { id: 'a' }])
  })
})

describe('reconcileOrderIds', () => {
  it('returns empty for no current ids', () => {
    expect(reconcileOrderIds([], ['a'])).toEqual([])
  })

  it('returns current ids when there is no saved order', () => {
    expect(reconcileOrderIds(['a', 'b'], [])).toEqual(['a', 'b'])
  })

  it('puts newly-seen ids ahead of the retained saved order', () => {
    expect(reconcileOrderIds(['fresh', 'a', 'b'], ['b', 'a', 'gone'])).toEqual(['fresh', 'b', 'a'])
  })
})

describe('sameIds', () => {
  it('is true only for identical ordered lists', () => {
    expect(sameIds(['a', 'b'], ['a', 'b'])).toBe(true)
    expect(sameIds(['a', 'b'], ['b', 'a'])).toBe(false)
    expect(sameIds(['a'], ['a', 'b'])).toBe(false)
  })
})
