<template>
  <section class="item">
    <button class="item__back" @click="$router.back()">← 返回推荐流</button>

    <div class="item__card">
      <ProductCard :id="itemId" :title="itemTitle" :cate-label="cateName" active />

      <div class="item__panel">
        <div class="item__id">
          <span class="item__label">当前商品</span>
          <span class="item__name">{{ itemTitle || `商品 #${itemId}` }}</span>
          <span class="num item__no">编号 {{ itemId }}<template v-if="cateName"> · {{ cateName }}</template></span>
          <span v-if="reasonLabel" class="item__reason">推荐理由 · {{ reasonLabel }}</span>
        </div>

        <div class="item__price" v-if="itemPrice !== null && itemPrice !== ''">
          <span class="item__price-sym">¥</span>{{ formatPrice(itemPrice) }}
          <span class="item__price-tag">演示价格 · 非数据集内容</span>
        </div>

        <div class="item__rate">
          <el-rate v-model="rating" show-text @change="onRate" />
          <p class="item__hint">打分以点击行为记录评分（1–5 星），画像即时刷新</p>
        </div>

        <div class="item__actions">
          <el-button @click="act('CART')">加入购物车</el-button>
          <el-button type="primary" class="btn-primary" @click="act('ORD')">立即购买</el-button>
          <el-button @click="act('FLW')">收藏</el-button>
        </div>

        <transition enter-active-class="msg-in">
          <p v-if="msg" class="item__msg" :class="{ 'is-err': msgIsErr }">{{ msg }}</p>
        </transition>
      </div>
    </div>

    <!-- 演示辅助：行为→画像→推荐 闭环可见（权威记录在后端 behavior 表）。
         真实电商不向终端用户暴露原始行为流水（UX 指南将"提醒用户被全程追踪"
         列为 surveillance-y 反模式，用户端只以"最近浏览/看了又看"呈现），
         此处仅为答辩演示：默认折叠 + 明确标注演示属性。 -->
    <el-collapse v-if="acts.length" class="item__acts">
      <el-collapse-item title="演示辅助 · 本次会话已记录的行为（行为 → 画像 → 推荐闭环）">
        <ul class="item__acts-list">
          <li v-for="(a, i) in acts" :key="i" class="item__acts-item">
            <span class="item__acts-time">{{ a.t }}</span>
            <span>{{ a.label }}</span>
          </li>
        </ul>
      </el-collapse-item>
    </el-collapse>

    <!-- 同类商品 · 看了又看（复用 /api/item/search 按类目取货，排除当前商品） -->
    <div v-if="related.length" class="item-related">
      <h3 class="item-related__title">同类商品 · 看了又看</h3>
      <el-row :gutter="12" class="item-related__grid">
        <el-col :xs="12" :sm="8" v-for="r in related" :key="r.item_id">
          <ProductCard
            :id="r.item_id"
            :cate-label="r.category_name || ''"
            :price="r.demo_price ?? null"
            :title="r.title || ''"
            @click="go(r.item_id)"
          />
        </el-col>
      </el-row>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { searchItems, sendFeedback } from '../api'
import ProductCard from '../components/ProductCard.vue'
import { formatPrice } from '../format_price'
import { REASON_LABELS } from '../reason_labels'
import { acts, pushAct } from '../session_acts'

const route = useRoute()
const router = useRouter()
// itemId 必须是 ref + watch：SPA 内切换商品（点"看了又看"、从清单页进详情）
// 会复用组件不重挂载，旧实现只读一次 params，导致页面数据、行为记录全部
// 停留在上一个商品（URL 与内容不一致，加购/收藏记错商品）。
const itemId = ref(Number(route.params.id))
const userId = Number(localStorage.getItem('shoprec_user_id'))
const rating = ref(0)
const msg = ref('')
const msgIsErr = ref(false)
const itemTitle = ref('')
const cateName = ref('')
const itemPrice = ref(null)
// 推荐理由：从 Feed 跳转时携带召回来源（?src=fused 等）；直接访问无来源则不展示
const reasonLabel = ref(REASON_LABELS[String(route.query.src || '')] || '')
// 同类商品楼层（看了又看）数据
const related = ref([])

// 标题/类目来自 /api/item/search（编号前缀可唯一命中）；失败静默回落编号展示
async function loadItem(id) {
  // 重置：避免上一个商品的数据在新商品加载期间残留
  itemId.value = id
  rating.value = 0
  msg.value = ''
  itemTitle.value = ''
  cateName.value = ''
  itemPrice.value = null
  related.value = []
  reasonLabel.value = REASON_LABELS[String(route.query.src || '')] || ''
  try {
    const { data } = await searchItems({ q: String(id), k: 5 })
    const hit = (data.items || []).find((x) => Number(x.item_id) === id)
    if (hit) {
      itemTitle.value = hit.title || ''
      cateName.value = hit.category_name || ''
      itemPrice.value = hit.demo_price ?? null
      await loadRelated(Number(hit.category_id))
    }
  } catch (e) {
    // 静默：详情页保留编号展示
  }
}

onMounted(() => loadItem(Number(route.params.id)))
// 路由参数变化（同组件复用）时重载，保证 URL/内容/行为记录三者一致
watch(() => route.params.id, (id) => loadItem(Number(id)))

// 同类商品：复用搜索接口按类目取货，排除当前商品，最多 6 件
async function loadRelated(categoryId) {
  if (!categoryId) return
  try {
    const { data } = await searchItems({ category_id: categoryId, k: 12 })
    related.value = (data.items || [])
      .filter((x) => Number(x.item_id) !== itemId.value)
      .slice(0, 6)
  } catch (e) {
    // 相关推荐失败静默：不影响主信息展示
  }
}

function say(text, isErr = false) {
  msg.value = text
  msgIsErr.value = isErr
}

// 行为枚举与后端 Literal["CLICK","CART","ORD","FLW"] 对齐：
// 加购 CART、下单 ORD、收藏 FLW；打分以 CLICK 记录评分。
async function act(type) {
  try {
    await sendFeedback({ user_id: userId, item_id: itemId.value, behavior_type: type })
    const label = { CART: '加入购物车', ORD: '购买', FLW: '收藏' }[type]
    pushAct(`${label}：${itemTitle.value || `商品 #${itemId.value}`}`)
    say(`已记录行为：${label}（画像已刷新）`)
  } catch (e) {
    say(e.response?.data?.detail || '行为记录失败', true)
  }
}

async function onRate(v) {
  try {
    await sendFeedback({ user_id: userId, item_id: itemId.value, behavior_type: 'CLICK', rating: v })
    pushAct(`打分 ${v} 星：${itemTitle.value || `商品 #${itemId.value}`}`)
    say(`已打分 ${v} 星（以点击行为记录评分，画像已刷新）`)
  } catch (e) {
    say(e.response?.data?.detail || '打分失败', true)
  }
}

// 相关推荐卡点击：同样记录一次点击后进入详情
async function go(id) {
  try {
    await sendFeedback({ user_id: userId, item_id: id, behavior_type: 'CLICK' })
    pushAct(`浏览：商品 #${id}`)
  } catch (e) {
    // 落库失败不阻塞跳转
  }
  router.push(`/item/${id}`)
}
</script>

<style scoped>
.item {
  max-width: 720px;
}
.item__back {
  background: none;
  border: none;
  color: var(--ink-soft);
  font-size: 13px;
  cursor: pointer;
  padding: 0 0 12px;
}
.item__back:hover {
  color: var(--brand-ink);
}

.item__card {
  display: flex;
  gap: 20px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow-card);
  overflow: hidden;
}

/* 色卡占 40% */
.item__card :deep(.prod-card) {
  flex: 0 0 40%;
  border: none;
  box-shadow: none;
  cursor: default;
  align-self: stretch;
}
.item__card :deep(.prod-card__swatch) {
  height: 100%;
  min-height: 180px;
}

.item__panel {
  flex: 1;
  padding: 24px;
  display: flex;
  flex-direction: column;
}

.item__id {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.item__label {
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}
.item__id .num {
  font-size: 28px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: 0.5px;
}
.item__name {
  font-size: 22px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.3;
  margin-top: 2px;
}
.item__no {
  font-size: 13px !important;
  font-weight: 400 !important;
  color: var(--ink-soft) !important;
  letter-spacing: 0 !important;
}

.item__rate {
  margin: 20px 0;
}
.item__rate :deep(.el-rate) {
  --el-rate-height: 28px;
}
.item__hint {
  margin: 6px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-faint);
}

.item__actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.item__actions .btn-primary {
  /* 立即购买 = 全站唯一交易 CTA，用电商暖色与玉绿品牌色区分层级 */
  --el-button-bg-color: var(--cta);
  --el-button-border-color: var(--cta);
  --el-button-hover-bg-color: var(--cta-hover);
  --el-button-hover-border-color: var(--cta-hover);
}

.item__msg {
  margin-top: 16px;
  font-size: 13px;
  color: var(--brand-ink);
}
.item__msg.is-err {
  color: var(--danger);
}

/* 推荐理由 chip：与 Feed 卡内 reason 标签同款配色 */
.item__reason {
  align-self: flex-start;
  margin-top: 6px;
  font-size: var(--fs-caption);
  color: #fff;
  /* 与卡片 reason chip 同源实底（白字 6.18:1），原半透明玉绿仅约 3.3:1 */
  background: var(--brand-ink);
  padding: 2px 10px;
  border-radius: 999px;
}

/* 演示价格大数字块：电商详情页第一视觉要素 */
.item__price {
  margin: 14px 0 4px;
  font-size: 28px;
  font-weight: 700;
  color: var(--cta);
  line-height: 1.1;
}
.item__price-sym {
  font-size: 16px;
  font-weight: 600;
  margin-right: 2px;
}
.item__price-tag {
  margin-left: 10px;
  font-size: var(--fs-caption);
  font-weight: 400;
  color: var(--ink-faint);
}

/* 演示辅助：行为流水折叠面板（el-collapse 自带卡片外观，这里只控间距） */
.item__acts {
  margin-top: 16px;
}
.item-related__title {
  margin: 0 0 10px;
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
}
.item__acts-list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.item__acts-item {
  display: flex;
  gap: 12px;
  font-size: 13px;
  color: var(--ink-soft);
  padding: 7px 0;
  border-bottom: 1px dashed var(--line);
}
.item__acts-item:last-child {
  border-bottom: none;
}
.item__acts-time {
  color: var(--ink-faint);
  flex: none;
}

/* 同类商品楼层 */
.item-related {
  margin-top: 20px;
}
.item-related__grid {
  row-gap: 4px;
}
.item-related__grid .el-col {
  margin-bottom: 16px;
}

.msg-in {
  animation: msgIn 0.2s ease;
}
@keyframes msgIn {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
}

@media (max-width: 560px) {
  .item__card {
    flex-direction: column;
  }
  .item__card :deep(.prod-card) {
    flex: none;
  }
  .item__card :deep(.prod-card__swatch) {
    min-height: 120px;
  }
}
</style>
