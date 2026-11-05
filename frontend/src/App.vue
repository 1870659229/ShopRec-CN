<template>
  <div class="app">
    <header class="topbar">
      <div class="topbar__inner">
        <span class="topbar__brand">
          <span class="topbar__dot" aria-hidden="true"></span>
          品智推
          <span class="topbar__tag">新品冷启动推荐</span>
        </span>
        <nav class="topbar__nav">
          <router-link to="/feed">推荐流</router-link>
          <router-link to="/dashboard">后台看板</router-link>
        </nav>

        <!-- 淘宝式右侧用户导航组。结构参考 GitHub 淘宝顶栏复刻实现
             （FE-star/Plan-2022 page-transformer/src/components/header.vue）：
             用户名 + 已买到的宝贝 + 我的淘宝▾ + 购物车(角标) + 收藏夹▾。
             与真实淘宝的差异：退出并入「我的淘宝」下拉，不单独占位。 -->
        <nav v-if="userId" class="topbar__user-nav">
          <span class="topbar__user" :title="`user_id ${userId}`">
            <el-icon><User /></el-icon>{{ username || `用户 ${userId}` }}
          </span>
          <router-link to="/orders" class="topbar__link topbar__link--bought">已买到的宝贝</router-link>
          <!-- 触发方式必须是 click + 触发元素必须是 <button>：
               原实现是 <span> + trigger="hover"。span 不可聚焦，hover 又只认鼠标
               ⇒ 键盘用户打不开菜单、触屏用户同样点不开，而「退出登录」是本菜单
               唯一的登出入口 ⇒ 这两类用户完全无法登出。
               改 <el-button>（原生可聚焦、Enter/Space 可唤起）+ trigger="click"
               后，键鼠触屏三条路径全部打通。 -->
          <el-dropdown trigger="click" class="topbar__dd">
            <el-button link class="topbar__link">
              我的淘宝<el-icon><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item @click="go('/orders')">已买到的宝贝</el-dropdown-item>
                <el-dropdown-item @click="go('/cart')">购物车</el-dropdown-item>
                <el-dropdown-item @click="go('/favorites')">收藏夹</el-dropdown-item>
                <el-dropdown-item divided @click="logout">退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
          <router-link to="/cart" class="topbar__link">
            <el-icon><ShoppingCart /></el-icon>购物车
            <span v-if="cartCount" class="topbar__badge num">{{ cartCount }}</span>
          </router-link>
          <router-link to="/favorites" class="topbar__link">
            <el-icon><Star /></el-icon>收藏夹
            <span v-if="favCount" class="topbar__badge num">{{ favCount }}</span>
          </router-link>
        </nav>
        <el-button
          v-else
          link
          class="topbar__logout"
          @click="go('/login')"
        >
          请登录
        </el-button>
      </div>
    </header>
    <main class="page">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { User, ArrowDown, ShoppingCart, Star } from '@element-plus/icons-vue'
import { clearUserId, clearUsername, setUsername, userId, username } from './auth'
import { fetchUserBehaviors, fetchUserInfo } from './api'

const router = useRouter()
const cartCount = ref(0)
const favCount = ref(0)

function go(path) {
  router.push(path)
}

function logout() {
  clearUserId()
  clearUsername()
  router.push('/login')
}

// 角标计数：只数去重后的商品数（与清单页 total 同源同口径）。
// 失败静默回落 0（不显示角标），不阻塞页面。
async function refreshCounts() {
  if (!userId.value) {
    cartCount.value = 0
    favCount.value = 0
    return
  }
  try {
    const [cart, fav] = await Promise.all([
      fetchUserBehaviors({ user_id: userId.value, type: 'CART', limit: 99 }),
      fetchUserBehaviors({ user_id: userId.value, type: 'FLW', limit: 99 })
    ])
    cartCount.value = cart.data.total
    favCount.value = fav.data.total
  } catch {
    /* 计数失败不阻塞页面 */
  }
}

// 旧 session（localStorage 无 username，如 10-15 之前登录的用户）补查一次
async function ensureUsername() {
  if (userId.value && !username.value) {
    try {
      const { data } = await fetchUserInfo(userId.value)
      setUsername(data.username)
    } catch {
      /* 用户已被删：静默，顶栏回落"用户 id"展示 */
    }
  }
}

onMounted(() => {
  ensureUsername()
  refreshCounts()
})
// 登录/退出切换用户时同步刷新用户名与角标
watch(userId, () => {
  ensureUsername()
  refreshCounts()
})
// SPA 路由跳转（详情页加购/收藏 → 清单页）后角标同步：
// App 是根组件不随路由重挂载，只 watch userId 会留下陈旧计数
watch(
  () => router.currentRoute.value.path,
  () => {
    refreshCounts()
  }
)
</script>

<style scoped>
.app {
  min-height: 100vh;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line);
}
.topbar__inner {
  max-width: var(--page-max);
  margin: 0 auto;
  padding: 0 20px;
  height: 56px;
  display: flex;
  align-items: center;
  gap: 24px;
}

.topbar__brand {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 17px;
  color: var(--ink);
  letter-spacing: 0.5px;
}
.topbar__dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 0 3px rgba(18, 160, 111, 0.18);
}
.topbar__tag {
  font-size: var(--fs-caption);
  font-weight: 500;
  color: var(--ink-soft);
  padding: 1px 8px;
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  margin-left: 4px;
}

.topbar__nav {
  display: flex;
  gap: 4px;
}
.topbar__nav a {
  color: var(--ink-soft);
  font-size: 14px;
  padding: 6px 12px;
  border-radius: var(--radius-sm);
  transition: background 0.15s ease, color 0.15s ease;
}
.topbar__nav a:hover {
  background: var(--bg);
  color: var(--ink);
}
.topbar__nav a.router-link-active {
  color: var(--brand-ink);
  font-weight: 600;
  background: rgba(18, 160, 111, 0.08);
}

/* 淘宝式右侧用户导航组 */
.topbar__user-nav {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 2px;
}
.topbar__user {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-right: 6px;
  font-size: 13px;
  font-weight: 600;
  color: var(--brand-ink);
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.topbar__link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--ink-soft);
  font-size: 13px;
  padding: 6px 10px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.topbar__link:hover {
  background: var(--bg);
  color: var(--ink);
}
.topbar__link.router-link-active {
  color: var(--brand-ink);
  font-weight: 600;
  background: rgba(18, 160, 109, 0.08);
}
/* el-dropdown 触发元素与链接同款外观。
   原来是 outline:none（当时它是个不可聚焦的 <span>，掩盖了"不能 Tab 到"的事实）；
   现在是真 <button>，必须给回可见焦点环，否则键盘用户不知道焦点在哪。 */
.topbar__dd {
  vertical-align: middle;
}
.topbar__dd .topbar__link:focus-visible {
  outline: 2px solid var(--brand);
  outline-offset: 1px;
}
/* 角标：电商惯例用暖色（与 --cta 一致），非中国股市涨跌语义 */
.topbar__badge {
  min-width: 17px;
  height: 17px;
  padding: 0 5px;
  border-radius: 999px;
  background: var(--cta);
  color: #fff;
  font-size: 11px;
  line-height: 17px;
  text-align: center;
}

.topbar__logout {
  margin-left: auto;
  font-size: 13px;
  color: var(--ink-soft);
}

.page {
  max-width: var(--page-max);
  margin: 0 auto;
  padding: 28px 20px 48px;
}

@media (max-width: 860px) {
  /* 窄屏：独立入口收进「我的淘宝」下拉，避免顶栏拥挤 */
  .topbar__link--bought {
    display: none;
  }
  .topbar__inner {
    gap: 12px;
  }
}

@media (max-width: 560px) {
  .topbar__tag {
    display: none;
  }
}
</style>
