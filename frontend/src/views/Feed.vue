<template>
  <section>
    <div class="feed-head">
      <h2 class="feed-head__title">为你推荐</h2>
      <p class="feed-head__sub">
        <template v-if="mode === 'feed'">
          共 <span v-if="loaded" class="num">{{ items.length }}</span> 个商品 · {{ feedSubtitle }}
        </template>
        <template v-else>
          搜索到 <span class="num">{{ results.length }}</span> 个商品
          <el-button link size="small" class="feed-search__back" @click="backToFeed">返回推荐</el-button>
        </template>
      </p>
    </div>

    <!-- 淘宝式搜索条：实时联想 + 类目筛选 + 搜索按钮 -->
    <div class="feed-search">
      <el-autocomplete
        v-model="q"
        class="feed-search__input"
        placeholder="搜商品标题 / 编号 / 类目名，如 牛仔裤 / 9301 / 服饰"
        clearable
        :fetch-suggestions="querySuggest"
        :trigger-on-focus="false"
        @select="onSelect"
        @keyup.enter="doSearch"
        @clear="backToFeed"
      >
        <template #default="{ item }">
          <div class="feed-search__sg">
            <span class="feed-search__sg-main">{{ item.label }}</span>
            <span class="feed-search__sg-sub num">{{ item.sub }}</span>
          </div>
        </template>
      </el-autocomplete>
      <el-select
        v-model="cateId"
        class="feed-search__cate"
        placeholder="全部类目"
        clearable
      >
        <el-option
          v-for="c in cates"
          :key="c.category_id"
          :label="`${c.category_name}（${c.item_count}）`"
          :value="c.category_id"
        />
      </el-select>
      <el-button type="primary" class="feed-search__btn" @click="doSearch">搜索</el-button>
    </div>

    <!-- 淘宝式分类导航条：点击即按类目浏览（复用 /api/item/search） -->
    <div v-if="cates.length" class="feed-cates">
      <button
        v-for="c in cates"
        :key="c.category_id"
        class="feed-cates__item"
        :class="{ 'is-active': cateId === c.category_id && mode === 'search' }"
        @click="pickCate(c.category_id)"
      >
        <el-icon class="feed-cates__icon"><component :is="cateIcon(c.category_id)" /></el-icon>
        <span class="feed-cates__name">{{ c.category_name }}</span>
        <span class="feed-cates__count num">{{ c.item_count }} 件</span>
      </button>
    </div>

    <el-skeleton v-if="busy || (!loaded && mode === 'feed')" :rows="4" animated />

    <template v-else>
      <el-row :gutter="16" class="feed-grid" v-if="shown.length">
        <el-col :xs="12" :sm="8" :md="6" v-for="it in shown" :key="it.item_id ?? it">
          <ProductCard
            :id="it.item_id ?? it"
            :cate-label="it.category_name || ''"
            :price="it.demo_price ?? null"
            :reason="REASON_LABELS[it.source] || ''"
            :title="it.title || ''"
            @click="go(it.item_id ?? it, it.source, it.title)"
            @expose="onExpose(it.item_id ?? it)"
          />
        </el-col>
      </el-row>

      <el-empty
        v-else
        :description="mode === 'search'
          ? '没搜到「' + lastQ + '」——可搜范围是商品标题、编号与类目名'
          : '暂无推荐，先去登录或产生一些行为试试'"
      />
    </template>

    <el-alert
      v-if="err"
      type="error"
      :title="err"
      :closable="false"
      show-icon
    />

    <!-- 数据红线④：展示层演示要素须标注"非数据集内容、不参与指标计算" -->
    <p class="feed-note">商品标题、示意图与价格为演示文案，非 JDsearch 数据集内容，不参与指标计算</p>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  Monitor,
  ShoppingBag,
  Brush,
  IceCreamRound,
  House,
  Basketball,
  Clock,
  View,
} from '@element-plus/icons-vue'
import { fetchCategories, fetchRecommend, searchItems, sendFeedback, suggestItems } from '../api'
import ProductCard from '../components/ProductCard.vue'
import { REASON_LABELS } from '../reason_labels'
import { pushAct } from '../session_acts'

// 类目图标（已用图标包实测存在）
const CATE_ICONS = {
  9301: Monitor,
  9302: ShoppingBag,
  9303: Brush,
  9304: IceCreamRound,
  9305: House,
  9306: Basketball,
  9307: Clock,
  9308: View,
}
const cateIcon = (cid) => CATE_ICONS[cid] || Monitor

const items = ref([])        // 推荐流（[{item_id, source}]）
const results = ref([])      // 搜索结果（含类目名，无 source → 不打原因标签）
const cates = ref([])        // 类目下拉/导航条
const q = ref('')
const lastQ = ref('')
const cateId = ref(null)
const mode = ref('feed')     // feed | search
const loaded = ref(false)
const busy = ref(false)
const err = ref('')
const router = useRouter()
const userId = Number(localStorage.getItem('shoprec_user_id'))

const shown = computed(() => (mode.value === 'search' ? results.value : items.value))

// 副标题随推荐状态切换。无画像（冷启动）时后端把召回来源统一改写为 "hot"，
// 有画像后为 fused / tgt / aux。此前文案写死为"冷启动走全局热门兜底"，
// 导致用户产生行为、推荐已个性化后仍显示兜底文案，答辩现场会被误读。
const isColdStart = computed(
  () => items.value.length > 0 && items.value.every((i) => i.source === 'hot'),
)
const feedSubtitle = computed(() => {
  if (err.value) return '加载失败'
  if (!loaded.value) return '加载中'
  if (!items.value.length) return '暂无推荐'
  return isColdStart.value ? '冷启动走全局热门兜底' : '个性化推荐 · 双路融合 + 跨品类迁移'
})

onMounted(async () => {
  try {
    const [rec, cats] = await Promise.all([
      fetchRecommend(userId, 40),
      fetchCategories().catch(() => ({ data: { categories: [] } })),
    ])
    items.value = rec.data.items
    cates.value = cats.data.categories
  } catch (e) {
    err.value = e.response?.data?.detail || '推荐加载失败，请稍后重试'
  } finally {
    loaded.value = true
  }
})

// 实时联想：类目 + 商品混排（淘宝式），类目在前
async function querySuggest(keyword, cb) {
  if (!keyword || !keyword.trim()) return cb([])
  try {
    const { data } = await suggestItems(keyword.trim())
    const sgs = [
      ...data.categories.map((c) => ({
        label: c.category_name,
        sub: `类目 · ${c.item_count} 件`,
        category_id: c.category_id,
      })),
      ...data.items.map((it) => ({
        label: it.title || `商品 #${it.item_id}`,
        sub: it.category_name || '',
        item_id: it.item_id,
      })),
    ]
    cb(sgs)
  } catch (e) {
    cb([])
  }
}

// 点联想条目：类目 → 选中该类目并搜索；商品 → 直接进详情（淘宝式）
function onSelect(sg) {
  if (sg.item_id) {
    go(sg.item_id, undefined, sg.label)
  } else if (sg.category_id) {
    pickCate(sg.category_id)
  }
}

// 点分类导航条：同联想里的类目路径
function pickCate(cid) {
  cateId.value = cid
  q.value = ''
  doSearch()
}

async function doSearch() {
  if (mode.value === 'feed' && !q.value.trim() && !cateId.value) return
  busy.value = true
  mode.value = 'search'
  lastQ.value = q.value.trim()
  try {
    const { data } = await searchItems({
      q: q.value.trim(),
      category_id: cateId.value ?? undefined,
      k: 200, // 演示库 120 商品全量返回（后端 le=200）；真实库场景由分页承接
    })
    results.value = data.items
  } catch (e) {
    err.value = e.response?.data?.detail || '搜索失败，请稍后重试'
  } finally {
    busy.value = false
  }
}

function backToFeed() {
  mode.value = 'feed'
  q.value = ''
  cateId.value = null
  results.value = []
}

// 曝光埋点：ProductCard 进视口（≥40% 可见）时触发，每商品每会话只记一次。
// 后端约定 EXPOSE 只落库不折叠画像（弱信号不稀释兴趣）。
const exposed = new Set()
async function onExpose(id) {
  if (!userId || exposed.has(id)) return
  exposed.add(id)
  try {
    await sendFeedback({ user_id: userId, item_id: Number(id), behavior_type: 'EXPOSE' })
  } catch (e) {
    // 曝光落库失败静默：不影响浏览体验
  }
}

// 进入详情即记一次点击曝光（pv → 后端枚举 CLICK）
async function go(id, source, title) {
  try {
    await sendFeedback({ user_id: userId, item_id: id, behavior_type: 'CLICK' })
    pushAct(`浏览：${title || `商品 #${id}`}`)
    ElMessage.success('已记录点击，画像即时刷新')
  } catch (e) {
    // 行为落库失败不阻塞跳转（如商品不在库中），仅提示
    err.value = e.response?.data?.detail || '行为记录失败'
  }
  // 携带召回来源供详情页展示「推荐理由」；直接访问/搜索进入无来源则不展示
  router.push(source ? { path: `/item/${id}`, query: { src: source } } : `/item/${id}`)
}
</script>

<style scoped>
.feed-head {
  margin-bottom: 16px;
}
.feed-head__title {
  margin: 0;
  font-size: var(--fs-hero);
  font-weight: 700;
  color: var(--ink);
}
.feed-head__sub {
  margin: 6px 0 0;
  font-size: 13px;
  color: var(--ink-soft);
}

.feed-search {
  display: flex;
  gap: 10px;
  margin-bottom: 14px;
}
.feed-search__input {
  flex: 1;
}
.feed-search__cate {
  width: 180px;
}
.feed-search__btn {
  --el-button-bg-color: var(--brand);
  --el-button-border-color: var(--brand);
}
.feed-search__back {
  margin-left: 8px;
}
.feed-search__sg {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.feed-search__sg-main {
  color: var(--ink);
}
.feed-search__sg-sub {
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}

.feed-cates {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin-bottom: 20px;
}
.feed-cates__item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 12px 6px 10px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  cursor: pointer;
  transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
  font: inherit;
}
.feed-cates__item:hover {
  transform: translateY(-2px);
  border-color: var(--line-strong);
}
.feed-cates__item.is-active {
  border-color: var(--brand);
  box-shadow: 0 0 0 3px rgba(18, 160, 111, 0.15);
}
.feed-cates__icon {
  font-size: 22px;
  color: var(--brand);
}
.feed-cates__name {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
}
.feed-cates__count {
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}

.feed-grid {
  row-gap: 4px;
}
.feed-grid .el-col {
  margin-bottom: 16px;
}

.feed-note {
  margin: 8px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-faint);
  text-align: center;
}

@media (max-width: 760px) {
  .feed-cates {
    grid-template-columns: repeat(3, 1fr);
  }
  .feed-search__cate {
    width: 130px;
  }
}
</style>
