from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken

from .serializers import RegisterSerializer, CustomTokenObtainPairSerializer, UserSerializer
from apps.portfolios.models import Portfolio, Allocation
from apps.strategies.models import Strategy


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            # Create default starting Paper Portfolio for the new user
            paper_port = Portfolio.objects.create(
                user=user,
                name="Portafolio Paper Inicial",
                kind="paper",
                base_currency="USD",
                paper_initial_capital=100000.00,
                trading_enabled=True,
            )

            # Assign default starter strategies if available
            strategies = Strategy.objects.filter(is_active=True)[:2]
            if strategies.count() >= 2:
                Allocation.objects.create(portfolio=paper_port, strategy=strategies[0], target_weight=0.5000)
                Allocation.objects.create(portfolio=paper_port, strategy=strategies[1], target_weight=0.5000)
            elif strategies.count() == 1:
                Allocation.objects.create(portfolio=paper_port, strategy=strategies[0], target_weight=1.0000)

            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            return Response({
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "username": user.username,
                    "is_staff": user.is_staff,
                    "is_superuser": user.is_superuser,
                    "timezone": user.timezone,
                    "plan": user.effective_plan.code,
                },
                "message": "Usuario registrado exitosamente."
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
