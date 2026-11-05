import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import './styles/theme.css'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import Login from './views/Login.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/login' },
    { path: '/login', component: Login },
    { path: '/feed', component: () => import('./views/Feed.vue') },
    { path: '/item/:id', component: () => import('./views/ItemDetail.vue') },
    { path: '/dashboard', component: () => import('./views/Dashboard.vue') },
    // 淘宝式用户清单：已买到的宝贝 / 购物车 / 收藏夹（参数化视图）
    { path: '/orders', component: () => import('./views/MyItems.vue'), props: { kind: 'orders' } },
    { path: '/cart', component: () => import('./views/MyItems.vue'), props: { kind: 'cart' } },
    { path: '/favorites', component: () => import('./views/MyItems.vue'), props: { kind: 'favorites' } }
  ]
})

// 未登录（本地无 user_id）访问受保护页面时跳回登录页
const PUBLIC_PATHS = new Set(['/login', '/'])
router.beforeEach((to) => {
  const hasUser = !!localStorage.getItem('shoprec_user_id')
  if (!PUBLIC_PATHS.has(to.path) && !hasUser) return '/login'
})

createApp(App).use(ElementPlus).use(router).mount('#app')
