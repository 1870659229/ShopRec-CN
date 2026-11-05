// 推荐原因：后端真实召回路 → 中文标签。只描述真实路径，不编 ItemCF 语义。
// fused=两路融合（重叠候选）、tgt=品类内召回、aux=跨品类迁移、hot=全局热门兜底。
// Feed.vue 与 ItemDetail.vue 共用（单一事实源，避免两处标签漂移）。
export const REASON_LABELS = {
  fused: '双路融合',
  tgt: '按你的兴趣',
  aux: '跨品类迁移',
  hot: '热门',
}
