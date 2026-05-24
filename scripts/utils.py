# =========================================================
# UTILITY FUNCTIONS
# =========================================================

# =========================================================
# PRINT SECTION HEADINGS
# =========================================================

def print_heading(title):

    print("\n" + "=" * 40)

    print(title)

    print("=" * 40)


# =========================================================
# PRINT SUCCESS MESSAGE
# =========================================================

def print_success(message):

    print(f"\n✅ {message}")


# =========================================================
# PRINT ERROR MESSAGE
# =========================================================

def print_error(message):

    print(f"\n❌ {message}")


# =========================================================
# PRINT WARNING MESSAGE
# =========================================================

def print_warning(message):

    print(f"\n⚠ {message}")


# =========================================================
# PRINT STEP TITLE
# =========================================================

def print_step(step, title):

    print("\n" + "#" * 50)

    print(f"STEP {step} - {title}")

    print("#" * 50)


# =========================================================
# DISPLAY DATAFRAME SHAPE
# =========================================================

def print_shape(dataframe):

    print("\n📊 Dataset Shape:")

    print(dataframe.shape)


# =========================================================
# DISPLAY FIRST ROWS
# =========================================================

def print_head(dataframe, rows=5):

    print(f"\n📄 First {rows} Rows:")

    print(dataframe.head(rows))