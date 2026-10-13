import streamlit as st
import pandas as pd
from database import init_db, create_ticket, get_all_tickets, update_ticket_status, delete_ticket
from ml_utils import predict_ticket_attributes

# Page config
st.set_page_config(
    page_title="AutoSupport - Ticket Center",
    page_icon="🎫",
    layout="wide"
)

# Initialize database
init_db()

# Navigation / Sidebar
st.sidebar.title("🎫 AutoSupport System")
menu = st.sidebar.radio(
    "Navigation",
    ["Customer Submission Form", "Admin Dashboard", "Performance Insights"]
)

# ------------------ Page 1: Customer Submission Form ------------------
if menu == "Customer Submission Form":
    st.title("📩 Submit a Support Ticket")
    st.markdown("Submit your request below. Our automated system will classify and prioritize it using Machine Learning.")

    with st.form("ticket_submission_form"):
        complaint_text = st.text_area(
            "Describe your issue in detail:",
            placeholder="Describe what happened, error messages, steps to reproduce..."
        )
        
        # Real-time ML Prediction
        pred_category = "Technical Support"
        pred_priority = "Medium"
        if complaint_text.strip():
            pred_category, pred_priority = predict_ticket_attributes(complaint_text)
            
        st.write("--- ")
        st.subheader("🧠 Auto-Classification Preview (AI Suggested)")
        col_left, col_right = st.columns(2)
        with col_left:
            st.info(f"**Predicted Category:** {pred_category}")
        with col_right:
            st.warning(f"**Predicted Priority:** {pred_priority}")
            
        submitted = st.form_submit_button("Submit Support Ticket")
        
        if submitted:
            if complaint_text.strip():
                ticket_id = create_ticket(complaint_text, pred_category, pred_priority)
                st.success(f"✅ Ticket successfully created! Ticket ID: #{ticket_id}")
            else:
                st.error("⚠️ Please enter a detailed description before submitting.")

# ------------------ Page 2: Admin Dashboard ------------------
elif menu == "Admin Dashboard":
    st.title("🕵️ Admin Support Dashboard")
    st.markdown("Filter, manage, update status, and resolve or delete submitted customer tickets.")

    tickets = get_all_tickets()
    if not tickets:
        st.info("No tickets submitted yet! Go to the 'Customer Submission Form' to submit one.")
    else:
        df = pd.DataFrame(tickets)
        
        # Filtering controls
        st.subheader("🔍 Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            categories = ["All"] + list(df['category'].unique())
            sel_cat = st.selectbox("Category", categories)
        with col2:
            priorities = ["All"] + list(df['priority'].unique())
            sel_prio = st.selectbox("Priority", priorities)
        with col3:
            statuses = ["All"] + list(df['status'].unique())
            sel_status = st.selectbox("Status", statuses)
            
        # Apply filters
        filtered_df = df.copy()
        if sel_cat != "All":
            filtered_df = filtered_df[filtered_df['category'] == sel_cat]
        if sel_prio != "All":
            filtered_df = filtered_df[filtered_df['priority'] == sel_prio]
        if sel_status != "All":
            filtered_df = filtered_df[filtered_df['status'] == sel_status]
            
        st.subheader(f"📋 Live Tickets ({len(filtered_df)} shown)")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Action Center for status management and deletion
        st.write("--- ")
        st.subheader("🛠️ Ticket Action Center")
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            st.markdown("### Update Status")
            ticket_id_to_update = st.number_input("Enter Ticket ID to update:", min_value=1, step=1)
            new_status = st.selectbox("Set status to:", ["Open", "In Progress", "Resolved"])
            if st.button("Apply Status Change"):
                success = update_ticket_status(ticket_id_to_update, new_status)
                if success:
                    st.success(f"Ticket #{ticket_id_to_update} successfully updated to '{new_status}'!")
                    st.rerun()
                else:
                    st.error(f"Could not find Ticket ID #{ticket_id_to_update}.")
                    
        with action_col2:
            st.markdown("### Delete Ticket")
            ticket_id_to_delete = st.number_input("Enter Ticket ID to delete:", min_value=1, step=1, key="delete_id")
            confirm_delete = st.checkbox("Confirm deletion")
            if st.button("Delete Ticket permanently", type="primary"):
                if confirm_delete:
                    success = delete_ticket(ticket_id_to_delete)
                    if success:
                        st.success(f"Ticket #{ticket_id_to_delete} successfully deleted!")
                        st.rerun()
                    else:
                        st.error(f"Could not find Ticket ID #{ticket_id_to_delete}.")
                else:
                    st.warning("Please confirm deletion by checking the checkbox.")

# ------------------ Page 3: Performance Insights ------------------
elif menu == "Performance Insights":
    st.title("📊 Performance & Volume Insights")
    st.markdown("Key metrics and visual distribution of customer complaints stored in the local SQLite database.")

    tickets = get_all_tickets()
    if not tickets:
        st.info("No data available to display charts. Submit tickets first.")
    else:
        df = pd.DataFrame(tickets)
        
        # Summary metric blocks
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Tickets", len(df))
        with col_b:
            open_count = len(df[df['status'] == 'Open'])
            st.metric("Open Tickets", open_count)
        with col_c:
            resolved_count = len(df[df['status'] == 'Resolved'])
            st.metric("Resolved Tickets", resolved_count)
            
        st.write("--- ")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("📂 Ticket Breakdown by Category")
            cat_counts = df['category'].value_counts()
            st.bar_chart(cat_counts)
            
        with chart_col2:
            st.subheader("⚡ Ticket Breakdown by Priority")
            prio_counts = df['priority'].value_counts()
            st.bar_chart(prio_counts)
