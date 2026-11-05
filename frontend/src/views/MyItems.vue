<template>
  <section class="myitems">
    <header class="myitems__head">
      <h2 class="myitems__title">{{ meta.title }}</h2>
      <span v-if="!busy" class="myitems__count num">{{ items.length }} 件</span>
    </header>
    <p class="myitems__sub">{{ meta.sub }}</p>

    <el-skeleton v-if="busy" :rows="4" animated />

    <template v-else>
      <el-row v-if="items.length" :gutter="16" class="myitems__grid">
        <el-col :xs="12" :sm="8" :md="6" v-for="it in items" :key="it.item_id">
          <ProductCard
            :id="it.item_id"
            :cate-label="it.category_name || ''"
            :price="it.demo_price ?? null"
            :title="it.title || ''"
            @click="router.push(`/item/${it.item_id}`)"
          />
        </el-col>
      </el-row>
      <el-empty v-else :description="meta.empty" />

      <p class="myitems__note">
        数据源：behavior 行为流水（{{ meta.type }}），按商品去重、最近一次行为在前；
        演示环境为 append-only 流水，暂不支持移出清单
      </p>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import ProductCard from '../components/ProductCard.vue'
import { fetchUserBehaviors } from '../api'
import { userId } from '../auth'

// 三个淘宝式用户清单共用一个参数化视图，路由以 props 注入 kind
const props = defineProps({
  kind: { type: String, required: true } // orders | cart | favorites
})

const router = useRouter()
const items = ref([])
const busy = ref(true)

const META = {
  orders: {
    title: '已买到的宝贝',
    sub: '你购买过的商品——已购商品不再进入推荐流',
    type: 'ORD',
    empty: '还没有购买过商品——去推荐流逛逛，在详情页点「立即购买」即可生成订单'
  },
  cart: {
    title: '购物车',
    sub: '已加入购物车的商品',
    type: 'CART',
    empty: '购物车是空的——在商品详情页点「加入购物车」'
  },
  favorites: {
    title: '收藏夹',
    sub: '收藏过的商品',
    type: 'FLW',
    empty: '收藏夹是空的——在商品详情页点「收藏」'
  }
}
const meta = computed(() => META[props.kind] || META.cart)

// 三个清单路由（/orders /cart /favorites）复用同一组件：hash 跳转不重载
// 页面，onMounted 不重跑——必须 watch props.kind 重取数，否则 /orders 的
// 数据会残留显示在 /favorites 上（与 ItemDetail 的 params 监听同一修复模式）。
async function load() {
  busy.value = true
  try {
    const { data } = await fetchUserBehaviors({
      user_id: userId.value,
      type: meta.value.type,
      limit: 60
    })
    items.value = data.items
  } catch {
    items.value = []
  } finally {
    busy.value = false
  }
}

onMounted(load)
watch(() => props.kind, load)
</script>

<style scoped>
.myitems__head {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.myitems__title {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: var(--ink);
}
.myitems__count {
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}
.myitems__sub {
  margin: 6px 0 18px;
  font-size: 13px;
  color: var(--ink-soft);
}
.myitems__grid {
  margin-top: 4px;
}
.myitems__note {
  margin-top: 22px;
  font-size: var(--fs-caption);
  color: var(--ink-faint);
}
</style>
