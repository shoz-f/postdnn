defmodule PostDNN do
  @moduledoc """
  Post-processing utilities for Deep Neural Network.
  """

  alias PostDNN.NIF

  @doc """
  Execute post processing: nms.

  ## Parameters

    * num_boxes       - number of candidate boxes
    * num_class       - number of category class
    * boxes           - binaries, serialized boxes tensor[`num_boxes`][4]; dtype: float32
    * scores          - binaries, serialized score tensor[`num_boxes`][`num_class`]; dtype: float32
    * opts
      * iou_threshold:   - IOU threshold
      * score_threshold: - score cutoff threshold
      * sigma:           - soft IOU parameter
      * boxrepr:         - type of box representation
         * 0 - center pos and width/height
         * 1 - top-left pos and width/height
         * 2 - top-left and bottom-right corner pos

  ## Examples

    ```elixir
    non_max_suppression_multi_class(
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores), boxrepr: :corner
    )
    ```
  """
  def non_max_suppression_multi_class({num_boxes, num_class}, boxes, scores, opts \\ []) do
    box_repr = case Keyword.get(opts, :boxrepr, :center) do
      :center  -> 0
      :topleft -> 1
      :corner  -> 2
    end

    iou_threshold   = Keyword.get(opts, :iou_threshold, 0.5)
    score_threshold = Keyword.get(opts, :score_threshold, 0.25)
    sigma           = Keyword.get(opts, :sigma, 0.0)

    case NIF.dnn_non_max_suppression_multi_class(num_boxes, box_repr, boxes, num_class, scores, iou_threshold, score_threshold, sigma) do
      {:ok, nil} -> :notfind
      {:ok, result} -> Poison.decode(result)
      any -> any
    end
  end

  @doc """
  Create a list of coordinates for mesh grid points - top-left of each grid.
  
  ## Parameters
  
    * shape - tupple {width, height} for overall size.
    * pitches - list of grid spacing.
    * opts
      * :center - return center of each grid.
  
  ## Examples
  
    ```
    mesh_grid({416,416}, [8,16,32,64], [:center])
    ```
  """
  def mesh_grid(shape, pitches, opts \\ [])

  def mesh_grid(shape, pitches, opts) when is_list(pitches) do
    Enum.map(pitches, &mesh_grid(shape, &1, opts))
    |> Nx.concatenate()
  end

  def mesh_grid({w, h}, pitch, opts) when w >= 1 and h >= 1 do
    m = trunc(Float.ceil(h/pitch))
    n = trunc(Float.ceil(w/pitch))

    # grid coodinates list
    grid = (for y <- 0..(m-1), x <- 0..(n-1), do: [x, y])
      |> Nx.tensor(type: {:f, 32})
      |> (&if :center in opts, do: Nx.add(&1, 0.5), else: &1).()
      |> Nx.multiply(pitch)

    # pitch list
    pitch = Nx.broadcast(pitch, {m*n, 1})
      |> Nx.as_type({:f, 32})

    Nx.concatenate([grid, pitch], axis: 1)
  end
end
