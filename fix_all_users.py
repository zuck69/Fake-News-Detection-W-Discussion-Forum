# fix_all_users.py
import sqlalchemy as sa
from app import db, create_app
from app.models import User

# Try to use create_app if you use an application factory layout
try:
    flask_app = create_app()
except TypeError:
    # If create_app requires configuration arguments, or if you use a module-level variable
    from app import app as flask_app

with flask_app.app_context():
    # 1. Fetch all users from your app.db file
    users = db.session.scalars(sa.select(User)).all()
    
    print("🚀 Scanning database to update passwords...")
    updated_count = 0
    
    for user in users:
        # Check if the user is using the broken scrypt hash format
        if user.password_hash and user.password_hash.startswith('scrypt:'):
            print(f"Updating password for user: '{user.username}'")
            
            # This sets the password to "user" using the supported pbkdf2 method
            user.set_password('user') 
            updated_count += 1
            
    # 2. Save all the updates to app.db
    if updated_count > 0:
        db.session.commit()
        print(f"\n🏁 Finished! Successfully updated {updated_count} user accounts.")
        print("🔒 Every existing user can now log in using the password: user")
    else:
        print("\n🏁 No users with 'scrypt' hashes were found to update.")
