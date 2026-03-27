import axios from 'axios'

const api = axios.create({
  baseURL: '',   // proxied by Vite → localhost:8000
  timeout: 60000,
})

// Attach JWT on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-logout on 401
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.clear()
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// ── Auth ──────────────────────────────────────────────────────────────────────

export const authService = {
  async register({ full_name, email, password, role = 'user' }) {
    const { data } = await api.post('/auth/register', { full_name, email, password, role })
    return data
  },

  async login({ email, password }) {
    const { data } = await api.post('/auth/login', { email, password })
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('user', JSON.stringify({
      id: data.user_id,
      full_name: data.full_name,
      role: data.role,
    }))
    return data
  },

  logout() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('user')
  },

  getCurrentUser() {
    const raw = localStorage.getItem('user')
    return raw ? JSON.parse(raw) : null
  },

  isAuthenticated() {
    return !!localStorage.getItem('access_token')
  },
}

// ── Prediction ────────────────────────────────────────────────────────────────

export const predictionService = {
  async predict(file) {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await api.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },

  async getReport(predictionData) {
    const { data } = await api.post('/predict/report', predictionData, {
      responseType: 'blob',
    })
    return data
  },
}

// ── History ───────────────────────────────────────────────────────────────────

export const historyService = {
  async list({ page = 1, pageSize = 10, tumorType = null } = {}) {
    const params = { page, page_size: pageSize }
    if (tumorType) params.tumor_type = tumorType
    const { data } = await api.get('/history', { params })
    return data
  },

  async getDetail(scanId) {
    const { data } = await api.get(`/history/${scanId}`)
    return data
  },

  async deleteScan(scanId) {
    await api.delete(`/history/${scanId}`)
  },

  async submitFeedback(scanId, feedback, notes = null) {
    const { data } = await api.post(`/history/${scanId}/feedback`, { feedback, notes })
    return data
  },
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export const analyticsService = {
  async getUserAnalytics() {
    const { data } = await api.get('/analytics/overview')
    return data
  },

  async getAdminStats() {
    const { data } = await api.get('/analytics/admin/platform')
    return data
  },
}

// ── Users ─────────────────────────────────────────────────────────────────────

export const userService = {
  async getProfile() {
    const { data } = await api.get('/users/me')
    return data
  },

  async updateProfile(payload) {
    const { data } = await api.patch('/users/me', payload)
    return data
  },

  async changePassword(currentPassword, newPassword) {
    const { data } = await api.post('/users/me/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
    return data
  },

  async listUsers({ page = 1, pageSize = 20, search = null } = {}) {
    const params = { page, page_size: pageSize }
    if (search) params.search = search
    const { data } = await api.get('/users', { params })
    return data
  },

  async deleteUser(userId) {
    await api.delete(`/users/${userId}`)
  },

  async changeUserRole(userId, newRole) {
    const { data } = await api.patch(`/users/${userId}/role`, null, { params: { new_role: newRole } })
    return data
  },
}

export default api
