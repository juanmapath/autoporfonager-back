from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from apps.users.models import Plan

User = get_user_model()


class PlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plan
        fields = ("id", "code", "name", "max_live_portfolios", "max_paper_portfolios", "max_broker_accounts_per_portfolio")


class UserSerializer(serializers.ModelSerializer):
    plan = PlanSerializer(read_only=True)

    class Meta:
        model = User
        fields = ("id", "email", "username", "is_staff", "is_superuser", "timezone", "plan", "date_joined")


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ("email", "username", "password", "timezone")
        extra_kwargs = {
            "username": {"required": False},
            "timezone": {"required": False},
        }

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("Ya existe una cuenta registrada con este correo electrónico.")
        return value.lower()

    def create(self, validated_data):
        email = validated_data["email"]
        username = validated_data.get("username") or email.split("@")[0]
        # Ensure unique username
        base_username = username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}_{counter}"
            counter += 1

        user = User.objects.create_user(
            email=email,
            username=username,
            password=validated_data["password"],
            timezone=validated_data.get("timezone", "America/Bogota"),
        )
        return user


class CustomTokenObtainPairSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        email_or_user = attrs.get("email", "").strip()
        password = attrs.get("password", "")

        user = (
            User.objects.filter(email__iexact=email_or_user).first()
            or User.objects.filter(username__iexact=email_or_user).first()
        )

        if not user or not user.check_password(password):
            raise serializers.ValidationError("Credenciales incorrectas. Verifica tu correo y contraseña.")

        if not user.is_active:
            raise serializers.ValidationError("Esta cuenta de usuario se encuentra inactiva.")

        from rest_framework_simplejwt.tokens import RefreshToken
        refresh = RefreshToken.for_user(user)
        return {
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
        }
