<template>
  <div
    ref="rootEl"
    class="prod-card"
    :class="{ 'is-active': active }"
    :style="{ '--swatch': swatch }"
    role="button"
    tabindex="0"
    title="进入即记一次点击"
    @click="$emit('click')"
    @keyup.enter="$emit('click')"
  >
    <div class="prod-card__swatch" :class="{ 'has-img': showImg }" aria-hidden="true">
      <img
        v-if="showImg"
        class="prod-card__img"
        :src="imgSrc"
        :alt="`商品 ${id} 示意图（非数据集内容，不参与指标计算）`"
        loading="lazy"
        decoding="async"
        @error="onImgError"
      />
      <span v-if="reason" class="prod-card__reason">{{ reason }}</span>
      <span class="prod-card__swatch-label">{{ cateLabel || '—' }}</span>
    </div>
    <div class="prod-card__body">
      <p class="prod-card__id">{{ title || `商品 #${id}` }}</p>
      <p v-if="priceText" class="prod-card__price">
        <span class="prod-card__price-sym">¥</span>{{ priceText }}
      </p>
      <p class="prod-card__sub">{{ subtitle || (title ? `商品 #${id}` : '暂无标签') }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { formatPrice } from '../format_price'

const props = defineProps({
  id: { type: [Number, String], required: true },
  cateLabel: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  // 演示标题：仅 demo 段位有值，真实数据为空 → 回落显示"商品 #编号"
  title: { type: String, default: '' },
  // 演示价格：仅 demo 段位有值（元）；真实数据为 null → 不展示价格行
  price: { type: [Number, String], default: null },
  active: { type: Boolean, default: false },
  // 推荐原因：来自后端真实召回路（fused/tgt/aux/hot 的中文映射），
  // 只描述真实召回路，不编造 ItemCF 语义。
  reason: { type: String, default: '' }
})

const priceText = computed(() => formatPrice(props.price))

const emit = defineEmits(['click', 'expose'])

// 曝光埋点：卡片进入视口（至少 40% 可见）即上报一次，每张卡生命周期内只发一次。
// IntersectionObserver 由 ProductCard 自持——Feed / 搜索结果 / 详情页复用组件
// 时，只有挂了 @expose 监听的地方（Feed）才真正产生上报。
const rootEl = ref(null)
let _observer = null
onMounted(() => {
  if (!('IntersectionObserver' in window) || !rootEl.value) return
  _observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        emit('expose')
        _observer.disconnect()
        _observer = null
      }
    },
    { threshold: 0.4 }
  )
  _observer.observe(rootEl.value)
})
onBeforeUnmount(() => {
  if (_observer) _observer.disconnect()
})

// 图源优先级：① 演示示意图 /items/<id>.（jpg|png）（非数据集内容，须标注示意）；
// ② 放图缺失 → 回落确定性色卡（按编号黄金角取色，同编号恒同色）。
// JDsearch 商品本身无图（item 表仅编号/品类/热度），此处图只用于演示观感。
//
// 为何用 fetch 探测而非纯 <img @error>：Vite dev 的 SPA fallback 对"不存在"的
// /items/x.jpg 也返回 200（内容是 index.html），img 标签因此不会 error，
// 会渲染出裂图。fetch 后可读 content-type：真图为 image/*，兜底页是 text/html。
// 顺序即优先级：2026-10-16 已把演示图统一重导出为 512×512 WebP（120/120，
// 总量 7.6MB→1.70MB），webp 先行保证一次探测即命中；jpg/png 保留在链尾兜底，
// 万一将来补入其他格式图仍可命中，探测失败最终回落确定性色卡。
//
// ★ 探测必须用 HEAD（2026-10-16 实测）：GET 会把整张图下载一遍只为读
//   content-type。HEAD 在 Vite dev 与 python -m http.server 上降级行为
//   与 GET 一致：缺失图 → Vite 200 text/html（SPA fallback）/ http.server
//   404，两种情况都被下方 `res.ok && ctype.startsWith('image/')` 挡下，
//   不存在探测失真。
const IMG_EXTS = ['webp', 'jpg', 'png']
const showImg = ref(false)
const imgSrc = ref('')

async function probeImg() {
  for (const ext of IMG_EXTS) {
    try {
      // 只读响应头：探测图片是否存在，不下载图片本体（见上方 ★ 注释）
      const res = await fetch(`/items/${props.id}.${ext}`, { method: 'HEAD' })
      const ctype = res.headers.get('content-type') || ''
      if (res.ok && ctype.startsWith('image/')) {
        imgSrc.value = `/items/${props.id}.${ext}`
        showImg.value = true
        return
      }
    } catch {
      // 网络异常：保持当前 ext 试下一个，最终回落色卡
    }
  }
  // 都未命中：保持色卡
  showImg.value = false
}
onMounted(probeImg)

// 兜底：fetch 命中但图片解码失败（如损坏）时隐藏，露出底层色卡
function onImgError() {
  showImg.value = false
}

// 签名：商品在数据中只有编号（JDsearch 无图）→ 按编号确定性取一色，
// 像一张"材质/口味卡"，同一编号永远同一色（可复现，不闪烁）。
const swatch = computed(() => {
  const n = Number(props.id) || 0
  const hue = (n * 137.508) % 360 // 黄金角，分布均匀
  const sat = 42 + ((n * 7) % 22) // 42–64%
  const light = 82 - ((n * 5) % 10) // 72–82%
  return `hsl(${hue.toFixed(1)} ${sat}% ${light}%)`
})
</script>

<style scoped>
.prod-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  overflow: hidden;
  cursor: pointer;
  box-shadow: var(--shadow-card);
  transition:
    transform 0.16s ease,
    box-shadow 0.16s ease,
    border-color 0.16s ease;
  outline: none;
}
.prod-card:hover,
.prod-card:focus-visible {
  transform: translateY(-3px);
  box-shadow: var(--shadow-hover);
  border-color: var(--line-strong);
}
.prod-card:focus-visible {
  box-shadow: 0 0 0 3px rgba(18, 160, 111, 0.35), var(--shadow-hover);
}
.prod-card.is-active {
  border-color: var(--brand);
  box-shadow: 0 0 0 3px rgba(18, 160, 111, 0.18), var(--shadow-card);
}

.prod-card__swatch {
  height: 150px;
  background: var(--swatch);
  position: relative;
}
/* 有示意图时铺满；加载失败或无图时回落确定性色卡（--swatch 仍保留在底层） */
.prod-card__img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.prod-card__reason {
  position: absolute;
  left: 10px;
  top: 10px;
  font-size: var(--fs-caption);
  color: #fff;
  /* 实底深玉绿（6.18:1）：原半透明玉绿叠在浅色卡上白字仅约 3.3:1，不达 AA；
     且半透明底色随下层色卡变化，对比度不可控。--brand-ink 与链接/强调同源 */
  background: var(--brand-ink);
  padding: 2px 8px;
  border-radius: 999px;
  max-width: 110px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prod-card__swatch-label {
  position: absolute;
  right: 10px;
  bottom: 10px;
  font-size: var(--fs-caption);
  color: var(--ink);
  background: rgba(255, 255, 255, 0.82);
  padding: 2px 8px;
  border-radius: 999px;
  backdrop-filter: blur(2px);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.prod-card__body {
  padding: 10px 12px 12px;
}
.prod-card__id {
  margin: 0;
  font-weight: 600;
  font-size: var(--fs-body);
  color: var(--ink);
  /* 两行截断：长标题完整可读，min-height 兜底单行标题保持网格行对齐 */
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 44px;
}
/* 演示价格：电商卡片三大必选要素（主图/标题/售价）之一，用 CTA 暖色 */
.prod-card__price {
  margin: 4px 0 0;
  font-size: 15px;
  font-weight: 700;
  color: var(--cta);
}
.prod-card__price-sym {
  font-size: 11px;
  font-weight: 600;
  margin-right: 1px;
}
.prod-card__sub {
  margin: 2px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}
</style>
