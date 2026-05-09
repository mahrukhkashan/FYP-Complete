from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from config.database_config import get_db_connection
from api.decorators import role_required

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/admin/create-user", methods=["POST"])
@role_required("admin")
def create_user():
    data = request.json
    role = data["role"]

    if role not in ["clinician", "attendant"]:
        return {"error": "Invalid role"}, 400

    db = get_db_connection()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO users (username,email,password_hash,role)
        VALUES (%s,%s,%s,%s)
    """, (
        data["username"],
        data["email"],
        generate_password_hash(data["password"]),
        role
    ))
    db.commit()
    cur.close()
    db.close()

    return {"success": True}
