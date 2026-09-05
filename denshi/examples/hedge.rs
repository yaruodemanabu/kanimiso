use denshi::Hedge;
fn main() -> denshi::Result<()> {
    let mut learner = Hedge::new(3, 0.5)?;
    for losses in [[0.0, 0.5, 1.0], [0.0, 1.0, 0.5]] {
        println!("mixture: {:?}", learner.probabilities());
        learner.update(&losses)?;
    }
    println!("external regret: {}", learner.regret());
    Ok(())
}
