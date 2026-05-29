from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check() -> dict[str, str]:
    return {"message": "ml churn service is running"}
