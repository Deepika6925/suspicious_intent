import pandas as pd

def log_frame_data(log, frame_id, score, motion, people, level):
    log.append({
        "Frame": frame_id,
        "Score": score,
        "Motion": motion,
        "People": people,
        "Level": level
    })
    return log

def save_metrics(log, filename="activity_log.csv"):
    df = pd.DataFrame(log)
    df.to_csv(filename, index=False)
    mean_score = df["Score"].mean()
    std_score = df["Score"].std()
    high_rate = (df["Level"]=="High").sum()/len(df)
    print(f"Frames: {len(df)}, Mean Score: {mean_score:.3f}, Std: {std_score:.3f}, High %: {high_rate*100:.1f}%")
    return df
