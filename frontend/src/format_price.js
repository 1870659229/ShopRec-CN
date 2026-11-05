// 演示价格格式化：卡片与详情页共用，保证两处展示一致。
// 价格是演示文案（seed_demo.PRICES，仅 9300 段有值），非数据集内容。
export function formatPrice(v) {
  if (v === null || v === undefined || v === '') return ''
  return Number(v).toLocaleString('zh-CN', { maximumFractionDigits: 2 })
}
