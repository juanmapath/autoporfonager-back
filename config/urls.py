from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.users.views import RegisterView, CustomTokenObtainPairView
from apps.portfolios.views import MeView, PortfolioViewSet, StrategyCatalogView
from apps.ops.views import (
    OpsOverviewView,
    OpsStrategiesView,
    OpsRecommendedPortfoliosView,
    OpsBacktestView,
    OpsKillSwitchView,
)

router = DefaultRouter()
router.register(r"portfolios", PortfolioViewSet, basename="portfolio")

urlpatterns = [
    path("admin/", admin.site.urls),

    # OpenAPI Schema & Docs
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),

    # Auth Endpoints
    path("api/v1/auth/register/", RegisterView.as_view(), name="auth_register"),
    path("api/v1/auth/login/", CustomTokenObtainPairView.as_view(), name="auth_login"),
    path("api/v1/auth/token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/v1/auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/v1/me/", MeView.as_view(), name="me"),

    # Public Strategy Catalog
    path("api/v1/strategies/", StrategyCatalogView.as_view(), name="public_strategies"),

    # User Portfolio Endpoints
    path("api/v1/", include(router.urls)),

    # Ops Admin Endpoints
    path("api/v1/ops/overview/", OpsOverviewView.as_view(), name="ops_overview"),
    path("api/v1/ops/strategies/", OpsStrategiesView.as_view(), name="ops_strategies"),
    path("api/v1/ops/recommended-portfolios/", OpsRecommendedPortfoliosView.as_view(), name="ops_recommended_portfolios"),
    path("api/v1/ops/backtests/", OpsBacktestView.as_view(), name="ops_backtests"),
    path("api/v1/ops/kill-switch/", OpsKillSwitchView.as_view(), name="ops_kill_switch"),
]
